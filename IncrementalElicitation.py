import csv
import numpy as np
from gurobipy import *

class CSS_Solver():

    def __init__(self, fichier_alternatives):
        self.A = dict()  # dictionnaire des points alternatives
        self.D_IdToMod = dict() # dictionnaire str -> int associant un entier a un nom de modele
        self.D_IdToCrit = dict() # dictionnaire str -> int associant un entier a un critere
        self.MMR_value = list()
        self.DM_W = None
        self.GurobiModel = None
        self.var_w = list()


        with open(fichier_alternatives) as csvfile:
            base_lignes_alternatives = csv.DictReader(csvfile, delimiter=',')
            L_criteres = list(set(base_lignes_alternatives.fieldnames) - {"md"})
            L_criteres.sort()
            self.L_criteres = L_criteres
            print(L_criteres)
            nb_criteres = len(L_criteres)
            self.M_Points = np.zeros((0, nb_criteres))
            self.D_IdToCrit = {i : L_criteres[i] for i in range(nb_criteres)}
            nb_alternatives = 0
            for ligne_alternative in base_lignes_alternatives:
                modele = ligne_alternative["md"]
                del ligne_alternative["md"]
                A_i = [int(ligne_alternative[criteria]) for criteria in L_criteres]
                self.M_Points = np.vstack((self.M_Points, np.array(A_i)))
                self.D_IdToMod[nb_alternatives] = modele
                nb_alternatives += 1
        # print(self.M_Points)
        self.i = self.ideal()
        self.n = self.nadir()
        dif_n_i = self.n - self.i
        self.M_Points = self.M_Points / dif_n_i        #NORMALISATION PAR L'ECART NADIR - IDEAL

        print("ideal", self.i)
        print("nadir", self.n)
        print(self.M_Points)


    def ideal(self):
        I = [min(self.M_Points[:,j]) for j in range(len(self.D_IdToCrit))]
        return np.array(I)

    def nadir(self):
        Non_Pareto_Points_id = set()

        for i1 in range(len(self.D_IdToMod)):
            for i2 in range(len(self.D_IdToMod)):
                if i1 != i2:
                    B = self.M_Points[i2, :] <= self.M_Points[i1,:]
                    # print(B)
                    nb_dominated_criteria = 0
                    for k in range(len(B)):
                        if B[k]:
                            nb_dominated_criteria += 1
                    if nb_dominated_criteria == len(self.D_IdToCrit):
                        Non_Pareto_Points_id.add(i1)
        Pareto_Points_id = {i for i in range(len(self.D_IdToMod))} - Non_Pareto_Points_id
        M_Pareto_Points = np.zeros((0, len(self.D_IdToCrit)))
        for i in Pareto_Points_id:
            M_Pareto_Points = np.vstack((M_Pareto_Points, self.M_Points[i, :]))
        N = [max(M_Pareto_Points[:,j]) for j in range(len(self.D_IdToCrit))]
        return np.array(N)


    def initialization(self, UnknownDMAgregationFunctionFile='DM_weights_file.csv'):
        with open(UnknownDMAgregationFunctionFile) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            W = ligne_poids.next()
            self.DM_W = [float(W[criteria]) for criteria in self.L_criteres]
            #Verification si strictement positifs et somme a 1
        self.GurobiModel = Model("MADMC")
        self.var_w = [self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name="w_%d"%num) for num in range(len(self.L_criteres))]
        self.GurobiModel.update()
        c1 = quicksum([w for w in self.var_w])  #contrainte c1 : sum{w_i} = 1
        self.GurobiModel.addConstr(c1 == 1)
        self.GurobiModel.update()


    def query(self):
        D_MR_yj = dict()               # dictionnnaire associant a chaque alternative i sa valeur MR et l'indice de son adversaire le pire

        for i in range(len(self.D_IdToMod)) :
            var_lambda = self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="lambda")
            self.GurobiModel.update()
            L_of_constraints_i = list()
            D_of_constraints_expr = dict()
            for j in range(len(self.D_IdToMod)):
                if (i != j):
                    #C_max_i_j : contraite de type Lambda >= f(x_i) - f(y_j)
                    C_max_i_j = LinExpr()
                    #f_x_i
                    f_x_i = quicksum(self.M_Points[i,k] * self.var_w[k] for k in range(len(self.L_criteres)))
                    f_y_j = quicksum(self.M_Points[j,k] * self.var_w[k] for k in range(len(self.L_criteres)))
                    C_max_i_j = var_lambda - f_x_i - f_y_j
                    D_of_constraints_expr[j] = C_max_i_j
                    L_of_constraints_i.append(self.GurobiModel.addConstr(C_max_i_j >= 0))
                    self.GurobiModel.update()
            self.GurobiModel.setObjective(var_lambda, GRB.MINIMIZE)
            self.GurobiModel.update()
            self.GurobiModel.optimize()
            MR_x_i = self.GurobiModel.objVal
            # j_star : indice du pire adversaire de i
            for j in D_of_constraints_expr:
                if D_of_constraints_expr[j].getValue() == MR_x_i:
                    j_star = j
            for constraint in L_of_constraints_i:
                self.GurobiModel.remove(constraint)
            self.GurobiModel.remove(var_lambda)
            self.GurobiModel.update()
            D_MR_yj[i] = (MR_x_i, j_star)

        #determiner MMR et donc la question à poser
        query_tuple = None
        MMR_value = None
        for i, MR_jStar in D_MR_yj.items():
            MR_i, j_star = MR_jStar
            if query_tuple == None or query_tuple[0] < MR_i:
                MMR_value = MR_i
                query_tuple = (i, j_star)
        return query_tuple

    def update_model_with_query(self, query):
        i, j = query
        f_x_i = sum([self.M_Points[i, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
        f_x_j = sum([self.M_Points[j, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
        f_x_i_expr = quicksum(self.M_Points[i,k] * self.var_w[k] for k in range(len(self.L_criteres)))
        f_y_j_expr = quicksum(self.M_Points[j,k] * self.var_w[k] for k in range(len(self.L_criteres)))
        if f_x_i <= f_x_j :
            self.GurobiModel.addConstr(f_x_i_expr - f_y_j_expr <= 0)
        else:
            self.GurobiModel.addConstr(f_y_j_expr - f_x_i_expr <= 0)
        self.GurobiModel.update()




if __name__ == '__main__':
    m = CSS_Solver('voitures.csv')
    m.initialization()

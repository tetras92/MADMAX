import csv
import numpy as np
from gurobipy import *

class CSS_Solver():

    def __init__(self, fichier_alternatives):
        self.A = dict()                    # dictionnaire des points alternatives
        self.D_IdToMod = dict()            # dictionnaire str -> int associant un entier a un nom de modele
        self.D_IdToCrit = dict()           # dictionnaire str -> int associant un entier a un critere
        self.MMR_values = list()           # Liste des MMR Valeurs calculees
        self.DM_W = None                   # vecteur de poids decrivant les preferences du decideur
        self.GurobiModel = None
        self.var_w = list()                # liste des variables poids a eliciter


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
        # print(self.M_Points)


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
        self.GurobiModel.setParam( 'OutputFlag', False)
        self.var_w = [self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name="w_%d"%num) for num in range(len(self.L_criteres))]
        self.GurobiModel.update()
        c1 = quicksum([w for w in self.var_w])  #contrainte c1 : sum{w_i} = 1
        self.GurobiModel.addConstr(c1 == 1)
        self.GurobiModel.update()


    def query(self):
        D_MR_yj = dict()               # dictionnnaire associant a chaque alternative i sa valeur MR et l'indice de son adversaire le pire

        for i in range(len(self.D_IdToMod)):
            L_PMR_i_j = list()
            for j in range(len(self.D_IdToMod)):
                if (i != j):
                    f_x_i = quicksum(self.M_Points[i,k] * self.var_w[k] for k in range(len(self.L_criteres)))
                    f_y_j = quicksum(self.M_Points[j,k] * self.var_w[k] for k in range(len(self.L_criteres)))
                    self.GurobiModel.setObjective(f_x_i - f_y_j, GRB.MAXIMIZE)
                    self.GurobiModel.update()
                    self.GurobiModel.optimize()
                    PMR_i_j = self.GurobiModel.objVal
                    L_PMR_i_j.append((PMR_i_j, j))


            # j_star : indice du pire adversaire de i
            L_PMR_i_j.sort(key=lambda c : c[0], reverse=True)
            MR_i , j_star = L_PMR_i_j[0]
            # if MR_i > L_PMR_i_j[1][0]:
            #     print("===>>Unique")
            # elif MR_i == L_PMR_i_j[1][0]:
            #     print("===>not Unique")

            D_MR_yj[i] = (MR_i, j_star)

        #determiner MMR et donc la question a poser
        query_tuple = None
        MMR_value = None

        for i, MR_jStar in D_MR_yj.items():
            MR_i, j_star = MR_jStar
            if query_tuple == None or MMR_value > MR_i:
                MMR_value = MR_i
                query_tuple = (i, j_star)

        print("Minimax Regret ", MMR_value)
        self.MMR_values.append(MMR_value)
        return query_tuple

    def update_model_with_query(self, query):
        i, j = query
        # for j in L_j:
        f_x_i = sum([self.M_Points[i, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
        f_x_j = sum([self.M_Points[j, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
        f_x_i_expr = quicksum(self.M_Points[i,k] * self.var_w[k] for k in range(len(self.L_criteres)))
        f_y_j_expr = quicksum(self.M_Points[j,k] * self.var_w[k] for k in range(len(self.L_criteres)))
        if f_x_i <= f_x_j :
            self.GurobiModel.addConstr(f_x_i_expr - f_y_j_expr <= 0)
            self.GurobiModel.update()
            print("DM prefers ", self.D_IdToMod[i], "to", self.D_IdToMod[j])
        else:
            self.GurobiModel.addConstr(f_y_j_expr - f_x_i_expr <= 0)
            self.GurobiModel.update()
            print("DM prefers ", self.D_IdToMod[j], "to", self.D_IdToMod[i])

        print("Contraint(s) added!\n\n")

    def DM_preference(self):
        i_star = None
        value_of_i_star = None
        for i in range(len(self.D_IdToMod)):
            f_i = sum([self.M_Points[i, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
            # print(self.D_IdToMod[i], f_i)
            if i_star == None or f_i < value_of_i_star:
                i_star = i
                value_of_i_star = f_i
        print("DM preferences using weights", self.D_IdToMod[i_star], value_of_i_star)
        return self.D_IdToMod[i_star]

    def best_alternative_elicitated(self):
        L_Var_x = list()
        L_f_expr_x = list()
        for i in range(len(self.D_IdToMod)):
            point_i = self.M_Points[i,:]
            L_Var_x.append(self.GurobiModel.addVar(vtype=GRB.BINARY, name="x_%d"%i))
            L_f_expr_x.append(quicksum(point_i[k]*self.var_w[k] for k in range(len(self.L_criteres))))
            self.GurobiModel.update()
        cst = self.GurobiModel.addConstr(quicksum(x for x in L_Var_x) == 1)
        self.GurobiModel.update()
        obj = quicksum(L_Var_x[i]*L_f_expr_x[i] for i in range(len(L_f_expr_x)))
        self.GurobiModel.update()
        self.GurobiModel.setObjective(obj, GRB.MINIMIZE)
        self.GurobiModel.optimize()
        i_p = None
        for i in range(len(self.D_IdToMod)):
            if L_Var_x[i].x == 1:
                i_p = i
        #retirer cst et les variables x
        self.GurobiModel.remove(cst)
        for x in L_Var_x:
            self.GurobiModel.remove(x)
        self.GurobiModel.update()

        return self.D_IdToMod[i_p]


    def start(self):
        nb = 0
        self.initialization()
        DM_preference_alternative = self.DM_preference()
        while len(self.MMR_values) == 0 or self.best_alternative_elicitated() != DM_preference_alternative:
            nb += 1
            query = self.query()
            print ("==============Query==============", query)
            self.update_model_with_query(query)

            # print("============================MMR_values===========================================", self.MMR_values)
        print(self.MMR_values)
        self.best_alternative_elicitated()



if __name__ == '__main__':

    m = CSS_Solver('voitures.csv')
    m.start()
    m.DM_preference()
    print(m.best_alternative_elicitated())


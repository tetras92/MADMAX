import csv
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import random

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
        # FOR RANDOM QUERY
        # self.Allqueries = list()
        # for q1 in range(len(self.D_IdToMod)):
        #     for q2 in range(q1+1, len(self.D_IdToMod)):
        #         self.Allqueries.append((q1, list([q2])))
        # random.shuffle(self.Allqueries)


    def generate_DM_random_vector(self):
        v = np.random.random_sample((len(self.L_criteres),))
        sum_ = np.sum(v)
        v /= sum_
        for i in range(len(self.L_criteres)):
            v[i] = round(v[i], 3)
        if np.sum(v) != 1.:
            v[0] = 1 - np.sum(v[1:])
        for i in range(len(self.L_criteres)):
            if v[i] == 0:
                return self.generate_DM_random_vector()
        while np.sum(v) != 1.:
            return self.generate_DM_random_vector()
        # print("random vector generated !!!")
        return v

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
                        # print("{} dominated by {}".format(self.D_IdToMod[i1], self.D_IdToMod[i2]))
                        Non_Pareto_Points_id.add(i1)
        # print({self.D_IdToMod[m] for m in Non_Pareto_Points_id})
        Pareto_Points_id = {i for i in range(len(self.D_IdToMod))} - Non_Pareto_Points_id
        M_Pareto_Points = np.zeros((0, len(self.D_IdToCrit)))
        for i in Pareto_Points_id:
            M_Pareto_Points = np.vstack((M_Pareto_Points, self.M_Points[i, :]))
        N = [max(M_Pareto_Points[:,j]) for j in range(len(self.D_IdToCrit))]
        return np.array(N)


    def initialization(self, UnknownDMAgregationFunctionFile='DM_weights_file.csv'):
        print("\n\nUploading DM linear agregation function from {} ...".format(UnknownDMAgregationFunctionFile))
        with open(UnknownDMAgregationFunctionFile) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            W = ligne_poids.next()
            self.DM_W = [float(W[criteria]) for criteria in self.L_criteres]

        # self.DM_W = self.generate_DM_random_vector()         #

        print("\t{}\n\t{}".format(self.L_criteres, self.DM_W))
        self.GurobiModel = Model("MADMC")
        self.GurobiModel.setParam( 'OutputFlag', False)
        self.var_w = [self.GurobiModel.addVar(vtype=GRB.CONTINUOUS, lb=0, name="w_%d"%num) for num in range(len(self.L_criteres))]
        self.GurobiModel.update()
        for w in self.var_w:
            self.GurobiModel.addConstr(w > 0)
        self.GurobiModel.update()
        c1 = quicksum([w for w in self.var_w])  #contrainte c1 : sum{w_i} = 1
        self.GurobiModel.addConstr(c1 == 1)
        self.GurobiModel.update()


    def query(self):
        D_MR_yj = dict()               # dictionnnaire associant a chaque alternative i sa valeur MR et la liste des indices de ses adversaires les pires

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
            L_j_star = list()
            L_PMR_i_j.sort(key=lambda c : c[0], reverse=True)       #ordre decroissant
            MR_i , j_star = L_PMR_i_j[0]
            L_j_star.append(j_star)
            it = 1
            while it < len(L_PMR_i_j) and L_PMR_i_j[it][0] == MR_i:
                L_j_star.append(L_PMR_i_j[it][1])
                it += 1

            D_MR_yj[i] = (MR_i, L_j_star)

        #determiner MMR et donc la question a poser
        query_tuple = None
        MMR_value = None

        for i, MR_jStar in D_MR_yj.items():
            MR_i, L_j_star = MR_jStar
            if query_tuple == None or MMR_value > MR_i:
                MMR_value = MR_i
                query_tuple = (i, L_j_star)
        nb_ = 0
        for i, MR_jStar in D_MR_yj.items():
            MR_i, L_j_star = MR_jStar
            if MR_i == MMR_value:
                nb_ += 1

        print("Minimax Regret alternative : {} [{}]".format(self.D_IdToMod[query_tuple[0]], round(MMR_value,3)))

        if nb_ > 1:
            print("=============================> At least 2 MMR")

        self.MMR_values.append(round(MMR_value,3))
        # query_tuple = self.Allqueries.pop(0)                             #random query
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
            print("\t\t  DM prefers {} to {}".format(self.D_IdToMod[i], self.D_IdToMod[j]))
        else:
            self.GurobiModel.addConstr(f_y_j_expr - f_x_i_expr <= 0)
            self.GurobiModel.update()
            print("\t\t  DM prefers {} to {}".format(self.D_IdToMod[j], self.D_IdToMod[i]))

        print("\t\t  Constraint added!")

    def DM_preference(self):
        i_star = None
        value_of_i_star = None
        for i in range(len(self.D_IdToMod)):
            f_i = sum([self.M_Points[i, k]*self.DM_W[k] for k in range(len(self.L_criteres))])
            # print(self.D_IdToMod[i], f_i)
            if i_star == None or f_i < value_of_i_star:
                i_star = i
                value_of_i_star = f_i
        print("\tDM preferences using weights : {}".format(self.D_IdToMod[i_star]))
        return self.D_IdToMod[i_star]


    def start(self):
        nb = 0
        self.initialization()
        DM_preference_alternative = self.DM_preference()
        print("================= ELICITATION STARTS! =================")
        while len(self.MMR_values) == 0 or self.MMR_values[-1] >= 0.0000001:

            query = self.query()
            x, L_j = query
            for x_ in L_j:
                print ("\t\t==============Query==============\n\t\t  {} vs. {}".format(self.D_IdToMod[x], self.D_IdToMod[x_]))
                self.update_model_with_query((x, x_))
                nb += 1

        print("================= ELICITATION STOPS! =================")
        print("DM alternative elicitated : {}".format(DM_preference_alternative))
        print("Number of queries : {}".format(nb))
        print("MMR values evolution : {}".format(self.MMR_values))
        plt.plot([i for i in range(1, len(self.MMR_values)+1)], self.MMR_values)
        plt.title("MMR values\n{}\n{}\n{}".format(self.L_criteres, self.DM_W, DM_preference_alternative))
        plt.show()
        # return nb, DM_preference_alternative                                                                                           # juste pour Q2a



if __name__ == '__main__':
    m = CSS_Solver('voitures.csv')
    m.start()


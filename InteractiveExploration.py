import csv
import numpy as np
class Modele:
    def __init__(self, fichier_alternatives):
        self.A = dict()                    # dictionnaire des points alternatives
        self.D_IdToMod = dict()            # dictionnaire str -> int associant un entier a un nom de modele
        self.D_IdToCrit = dict()           # dictionnaire str -> int associant un entier a un critere

        self.DM_W = None                   # vecteur de poids decrivant les preferences du decideur
        self.W = None                      # vecteur poids des criteres utilises

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
        self.compute_ideal_nadir()


        self.Still_available_alternatives = {i for i in range(len(self.D_IdToMod))}
        self.current_proposition = None                 # proposition courante

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

    def upload_criteria_weight(self, weight_file="weight_file.csv"):

        with open(weight_file) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            D_weight = ligne_poids.next()
            self.W = np.array([float(D_weight[criteria]) for criteria in self.L_criteres])
        print("Uploading weights to use from {}...\n\t{}\n\t{}".format(weight_file, self.L_criteres, self.W))


    def WA_Tchebycheff_norm(self, point_id, reference_point=None, epsilon=0.1):
        if reference_point == None:
            reference_point = self.i
        else:
            reference_point = np.array(reference_point)
        L = list()
        sum_ = 0                 # second membre  partie augmentee de la norme de Tchebycheff
        for i in range(len(self.L_criteres)):
            d_i = self.W[i] * (self.M_Points[point_id, i] - reference_point[i]) / self.dif_n_i[i]
            sum_ += d_i
            L.append(d_i)

        # print(L)
        return max(L) + epsilon * sum_


    def nearest_point_id(self, reference_point=None, epsilon=0.1):
        if reference_point != None:
            print("Reference Vector : \n\t{}\n\t{}".format(self.L_criteres, np.array(reference_point)))
        nearest_alternative_id = None
        shortest_distance = None
        for still_available_i in self.Still_available_alternatives:
            d = self.WA_Tchebycheff_norm(still_available_i, reference_point, epsilon)
            if nearest_alternative_id == None or shortest_distance > d:
                nearest_alternative_id = still_available_i
                shortest_distance = d
        print("\nProposition to DM : {}\n\t{}\n\t{}".format(self.D_IdToMod[nearest_alternative_id],
                                                          self.L_criteres, self.M_Points[nearest_alternative_id]))
        self.current_proposition = nearest_alternative_id
        return nearest_alternative_id


    def set_criteria_to_improve(self, criteria_id=2): #2 = euros
        value_of_this_criteria_in_current_opt = self.M_Points[self.current_proposition][criteria_id]
        # rappel : minimization
        self.Still_available_alternatives = {i for i in self.Still_available_alternatives if self.M_Points[i, criteria_id] < value_of_this_criteria_in_current_opt}
        #maj des points
        for i in range(len(self.D_IdToMod)):
            if i not in self.Still_available_alternatives:
                for k in range(len(self.L_criteres)):
                    self.M_Points[i, k] = float("inf")
        self.compute_ideal_nadir()

    def compute_ideal_nadir(self):
        self.i = self.ideal()
        self.n = self.nadir()
        # print("ideal", self.i)
        # print("nadir", self.n)
        self.dif_n_i = self.n - self.i

    def upload_DM_weight_preference(self, UnknownDMAgregationFunctionFile='DM_weights_file.csv'):
        print("Uploading DM linear agregation function from {} ...".format(UnknownDMAgregationFunctionFile))
        with open(UnknownDMAgregationFunctionFile) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            W = ligne_poids.next()
            self.DM_W = np.array([float(W[criteria]) for criteria in self.L_criteres])

        self.DM_prefered_alternative = None
        val_min = None
        for i in range(len(self.D_IdToMod)):
            val = np.sum((self.M_Points[i, :] * self.DM_W) / self.dif_n_i)
            if self.DM_prefered_alternative == None or val_min > val:
                self.DM_prefered_alternative = i
                val_min = val
        print("\tDM unknown preference is {} :\n\t{}\n\t{}".format(self.D_IdToMod[self.DM_prefered_alternative], self.L_criteres,
                                                                   self.M_Points[self.DM_prefered_alternative]))

    def criteria_to_improve(self):
        dif = (self.M_Points[self.current_proposition, :] - self.M_Points[self.DM_prefered_alternative,:]) / self.dif_n_i
        criteria_id  = 0
        max_value = dif[0]
        for i in range(1, len(self.L_criteres)):
            if dif[i] > max_value:
                max_value = dif[i]
                criteria_id = i
        print("\t{} {} is to high for DM !\n\n".format(self.M_Points[self.current_proposition, criteria_id],
                                                   self.L_criteres[criteria_id]))
        return criteria_id

    def start_exploration(self):
        self.upload_criteria_weight()
        self.upload_DM_weight_preference()

        while True:
            self.nearest_point_id()

            if self.current_proposition == self.DM_prefered_alternative:
                print("Exploration stopped : DM preference found!!!")
                return

            criteria_to_improve = self.criteria_to_improve()
            self.set_criteria_to_improve(criteria_to_improve)

    def nearest_alternative_to_a_reference_point(self, weight_file="weight_file.csv", performance_file="performance_cible.csv", epsilon=0.1):
        self.upload_criteria_weight(weight_file)
        with open(performance_file) as csvfile:
            ligne_performance = csv.DictReader(csvfile, delimiter=',')
            P = ligne_performance.next()
            reference_point = [float(P[criteria]) for criteria in self.L_criteres]
        self.Still_available_alternatives = {i for i in range(len(self.D_IdToMod))}                                     # securite
        n_p = self.nearest_point_id(reference_point=reference_point, epsilon=epsilon)
        return n_p

if __name__ == '__main__':
    m = Modele('voitures.csv')
    # m.start_exploration()
    m.nearest_alternative_to_a_reference_point()

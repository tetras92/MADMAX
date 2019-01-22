import csv
class Modele:

    def __init__(self, fichier_alternatives):
        self.A = dict() # dictionnaire des points alternatives
        self.I = dict() # point ideal
        self.N = dict() # point nadir approche
        self.X_star = dict()  # Dictionnaire : critere -> alternative optimisant ledit critere. Utilise pour la determination du nadir approche.
        with open(fichier_alternatives) as csvfile:
            self.base_lignes_alternatives = csv.DictReader(csvfile, delimiter=',')
            self.initialize_I_N_X_star()
            for ligne_alternative in self.base_lignes_alternatives:
                modele = ligne_alternative["md"]
                del ligne_alternative["md"]
                self.A[modele] = {f : int(value) for f, value in ligne_alternative.items()}


            self.E = self.A.keys()

    def initialize_I_N_X_star(self):
        self.I = {fieldname: None for fieldname in set(self.base_lignes_alternatives.fieldnames) - {"md"}}
        self.X_star = {fieldname: None for fieldname in set(self.base_lignes_alternatives.fieldnames) - {"md"}}
        self.N = {fieldname: list() for fieldname in set(self.base_lignes_alternatives.fieldnames) - {"md"}}

    def compute_I_and_N(self):
        self.initialize_I_N_X_star()
        for alternative_id in self.E:
            for fieldname in self.X_star:
                if self.I[fieldname] == None or self.I[fieldname] > int(self.A[alternative_id][fieldname]):
                    self.I[fieldname] = self.A[alternative_id][fieldname]
                    self.X_star[fieldname] = alternative_id

        #calcul du point nadir approche
        for fieldname, x in self.X_star.items():
            for f, value in self.A[x].items():
                self.N[f].append(value)
        self.N = {fieldname : max(L) for fieldname, L in self.N.items()}


    def upload_criteria_weight(self, weight_file):
        with open(weight_file) as csvfile:
            ligne_poids = csv.DictReader(csvfile, delimiter=',')
            self.W = ligne_poids.next()
            self.W = {f : float(value) for f,value in self.W.items()}


    def nearest_alternative_to_I(self, weight_file="weight_file.csv", epsilon=0.1):
        self.compute_I_and_N()
        self.upload_criteria_weight(weight_file=weight_file)
        #--- CALCUL DU VECTEUR LAMBDA ---#
        Lambda = dict()
        for fieldname, nadValue in self.N.items():
            Lambda[fieldname] = self.W[fieldname]/(nadValue - self.I[fieldname])
        #-------------------------------------------#
        #---DETERMINATION DE LA MEILLEURE SOLUTION---#

        L = list() #liste des couples modele, valeur de la norme de Tchebycheff
        for modele, composantes in self.A.items():
            if modele in self.E:
                max_value = None
                noise_expr = 0
                for i, x_i in composantes.items():
                    value = Lambda[i]*(x_i - self.I[i])
                    if max_value == None or value > max_value:
                        max_value = value
                    noise_expr += value


                L.append((modele, max_value + epsilon * noise_expr))
        L.sort(key=lambda c:c[1])
        print(L)
        self.O = L[0][0]        # solution optimale
        return L[0]

    def set_criteria_to_improve(self, criteria="eu"):
        value_of_this_criteria_in_current_opt = self.A[self.O][criteria]
        #rappel : minimization
        self.E = {a for a, composantes in self.A.items() if composantes[criteria] < value_of_this_criteria_in_current_opt}

    def upload_performance_vector(self, file="performance_cible.csv"):
        with open(file) as csvfile:
            ligne_performance = csv.DictReader(csvfile, delimiter=',')
            self.V_P = ligne_performance.next()
            self.V_P = {f : float(value) for f,value in self.V_P.items()}

    def nearest_alternative_to_V_P(self, weight_file="weight_file.csv", performance_file="performance_cible.csv", epsilon=0.1):
        self.E = self.A.keys()  #Reconsideration de l'ensemble des alternatives
        self.upload_performance_vector(file=performance_file)
        print(self.V_P)
        self.compute_I_and_N()
        # print(self.N)
        self.upload_criteria_weight(weight_file=weight_file)
        # --- CALCUL DU VECTEUR LAMBDA ---#
        Lambda = dict()
        for fieldname, nadValue in self.N.items():
            Lambda[fieldname] = self.W[fieldname] / (nadValue - self.I[fieldname])
        # -------------------------------------------#
        # ---DETERMINATION DE LA MEILLEURE SOLUTION---#
        L = list()  # liste des couples modele, valeur de la norme de Tchebycheff
        for modele, composantes in self.A.items():
            if modele in self.E:
                max_value = None
                noise_expr = 0
                for i, x_i in composantes.items():
                    value = Lambda[i] * (x_i - self.V_P[i])
                    if max_value == None or value > max_value:
                        max_value = value
                    noise_expr += value

                L.append((modele, max_value + epsilon * noise_expr))
        L.sort(key=lambda c: c[1])
        print(self.X_star)
        self.PP = L[0][0]  #solution la plus proche du vecteur de performance fourni
        print(self.A[self.PP])
        return L[0]

if __name__ == '__main__':
    m = Modele('voitures.csv')
    # m.compute_I_and_N()
    # print(m.I, m.N)
    # print("Ensemble des alternatives", m.E)
    print(m.nearest_alternative_to_I(weight_file="weight_file.csv", epsilon=0.1))
    # m.set_criteria_to_improve(criteria="cv")
    # print("Ensemble des alternatives", m.E)
    # print(m.nearest_alternative_to_I(weight_file="weight_file.csv", epsilon=0.1))
    # print(m.nearest_alternative_to_V_P())

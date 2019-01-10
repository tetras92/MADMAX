import csv

class CSS_Solver():

    def __init__(self, fichier_alternatives):
        self.A = dict()  # dictionnaire des points alternatives
        self.D_modToId = dict() #dictionnaire str -> int associant un entier a un nom de modele
        self.S_critToId = dict() #dictionnaire

        with open(fichier_alternatives) as csvfile:
            base_lignes_alternatives = csv.DictReader(csvfile, delimiter=',')

import csv

A = dict() # dictionnaire des points alternatives
L = list() #Liste des voitures
I = dict() #point ideal
N = dict() #point nadir approche

epsilon = 0.1

# #---CHARGEMENT DE FICHIER AVEC CALCUL DU POINT NADIR (APPROCHE) ET IDEAL---#
# -----------------------------------------------------------------------#
with open('voitures.csv') as csvfile:
    base_lignes_voiture = csv.DictReader(csvfile, delimiter=',')
    I = {fieldname : None for fieldname in set(base_lignes_voiture.fieldnames) - {"md"}}
    X_star = {fieldname : None for fieldname in set(base_lignes_voiture.fieldnames) - {"md"}}
    N = {fieldname : list() for fieldname in set(base_lignes_voiture.fieldnames) - {"md"}}
    for voiture in base_lignes_voiture:
        modele = voiture["md"]
        del voiture["md"]
        A[modele] = {f : int(value) for f, value in voiture.items()}

        # calcul point ideal
        for fieldname in X_star:
            if I[fieldname] == None or I[fieldname] > int(voiture[fieldname]):
                I[fieldname] = A[modele][fieldname]
                X_star[fieldname] = modele
    #calcul du point nadir approche
    for fieldname, x in X_star.items():
        for f, value in A[x].items():
            N[f].append(value)
    N = {fieldname : max(L) for fieldname, L in N.items()}


print("Nadir" , N)
print("IDEAL", I)

#-----------------------------------------------------------------------#
#---CALCUL DE LA SOLUTION LA PLUS PROCHE DU POINT IDEAL DANS LA DIRECTION DU POINT NADIR---#
W = dict() #dictionnaire des poids de criteres
with open('weight_file.csv') as csvfile:
    ligne_poids = csv.DictReader(csvfile, delimiter=',')
    W = ligne_poids.next()
    W = {f : float(value) for f,value in W.items()}
print("vecteur de poids", W)

#----------------------------------------------------------------------#
#--- CALCUL DU VECTEUR LAMBDA ---#
Lambda = dict()
for fieldname, nadValue in N.items():
    Lambda[fieldname] = W[fieldname]/(nadValue - I[fieldname])
print("Lambda ", Lambda)
#--------------------------------------------------------------------#
#---DETERMINATION DE LA MEILLEURE SOLUTION---#

L = list() #liste des couples modele, valeur de la norme de Tchebycheff
for modele, composantes in A.items():
    max_value = None
    noise_expr = 0
    for i, x_i in composantes.items():
        value = Lambda[i]*abs(x_i - I[i])
        if max_value == None or value > max_value:
            max_value = value
        noise_expr += value


    L.append((modele, max_value + epsilon * noise_expr))


L.sort(key=lambda c:c[1])
print(L[0][0], A[L[0][0]])

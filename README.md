# Maison du Monde
## Test technique Cécile Hannotte
Ce test comporte les livrables et leurs codes respectifs pour les parties Machine Learning, API et SQL.
L'analyse du dataset et le choix des différents paramètres de la partie ML sont expliqués dans le pdf `Maison_du_Monde_Test_Technique.pdf`.
<br>
Les requêtes SQL ont été effectuées dans le fichier `requete.sql`. En commentaire de ce fichier, un échantillon de la base de donnée a été recréé pour permettre les tests.

## Installation du modèle de prédiction
Installation des dépendances avec pip pour Windows et Linux :
`pip install -r pip_requirements.txt`
Installation avec conda sur Windows du fichier requierements.txt :
`conda create --name myenv --file requierements.txt`
<br>
Cette commande créera un environnement "myenv" avec toutes les packages nécessaires pour installer le projet.
## Lancement de l'API :
Le modèle utilise une API (flask) qui permet d'appeler directement les fonctions de prédictions dans `network.py`.
Tout d'abord se placer dans le projet, puis lancer l'api :
`cd test_technique/api 
python api.py`
<br>
Dans le navigateur sur le port 5000 (localhost) indiquer les dimensions du produit en suivant la route predict. 
Exemple, pour prédire la catégorie d'un objet de dimension 10x10x10 et de poids 10, rendez-vous à l'adresse :
`127.0.0.1:5000/predict?height=10&weight=10&width=10&depth=10`
<br>
Le serveur donnera sa réponse sous la forme d'une catégorie.

## SQL
Les requêtes sont créées dans un seul fichier. Un exemple de base de données a été créé pour permettre un test rapide sur la base. Pour utiliser cet exemple, il suffit de décommenter les lignes avant les requêtes. Elles permetteront la création des deux tables.
- La première requête renvoie pour chaque jour, le chiffre d'affaire enregistré dans une colonne "CA". Ce chiffre d'affaire est calculé selon le rapport quantité*prix_unité.
- La deuxième requête, effectue une jointure sur les deux tables. Elle permet de relier les tables par rapport à l'identifiant du client. La colonne "CAM" sera les ventes meubles et "CAD" sont les quantités de ventes décos.


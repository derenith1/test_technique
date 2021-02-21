# Installation du modèle de prédiction
Installation des dépendances avec pip pour Windows et Linux :
`pip install -r pip_requirements.txt`
Installation avec conda sur Windows du fichier requierements.txt.
## Pour lancer le programme :

`cd test_technique/api`
`python api.py`

Dans le navigateur sur le port 5000 (localhost) indiquer les dimensions du produit en suivant la route predict. Exemple :
`127.0.0.1:5000/predict?height=10&weight=10&width=10&depth=10`
Le serveur donnera sa réponse sous la forme d'une catégorie

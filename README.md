
## Słowem wstępu...

Nazywam się Marcin Rybiński i chciałbym bardzo podziękować 
za wzięcie pod uwagę mojej osoby i ciesze się, 
że mogę wziąć udział w kolejnym etapie rekrutacyjnym 
do Sotrender. Tyle słowem wstępu i zapraszam do 
mojego sposobu na rozwiązanie zadania 
[PolEval 2019 Cyberbullying detection - zadanie 6.2](http://2019.poleval.pl/index.php/tasks/task6). 

Rozwiązanie problemu rozpoczynam w skrypcie `1. analysis.ipynb`, w którym
przedstawiam analizę tweetów oraz budowa pierwszych prototypów. Następnie 
w skrypcie `2. train.ipynb` przedstawiam pomysł na ML Pipelin do uczenia i eksperymentowania
z omówieniem zagadnień CI/CD. Nauczone w ten sposób modele wdrażam w skrypcie 
`3. deployment.ipynb` korzystając z Dockera oraz Kubernetesa.

## Opis zawartości projektu

Główne skrypty z treścią analizy:

1. `1. analysis.ipynb` - Główny skrypt z analizą tweetów oraz prototypowanie.
2. `2. train.ipynb` - ML Pipeline + CI/CD
3. `3. deployment.ipynb` - Wdrożenie Docker + Kubernetes

Skrypty poboczne z funkcjami/klasami - używane wewnątrz głównych skryptów:

1. `preprocessing.py` - funkcje oraz klasy do przygotowania tekstu 

2. `validation.py` - funkcje do oceny i zwalidowania modeli

3. `visualizations.py` - funkcje do wizualizacji danych tekstowych

4. `app.py` - aplikacja we Flasku wystawiająca API do odpytania modelu

5. `train.py` - ML Pipeline w MLflow

6. `Dockerfile` - konfiguracja środowiska dla Dockera

7. `requirements.txt` - wymagane biblioteki

Foldery:

1. `data` - Dane do uczenia i testowania

2. `production` - Imitacja produkcji - zachowany model, który działa/będzie wdrażany

3. `mlruns` - Imitacja przeprowadzania eksperymentów i uczenia modeli - miejsce zapisu wyników z `train.py`
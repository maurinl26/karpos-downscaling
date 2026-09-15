# Contrastes climatiques locaux et QDM

## Deux usages distincts (#127)

- `EmpiricalQuantileMapping` : QM empirique, transfert historique fixe, queues
  bornées par défaut (`extrapolation="linear"` disponible).
- `WeatherQuantileMapping` : QM météo conservant le comportement des calibrateurs
  historiques, y compris les corrections aux queues et l'identité des mois non
  calibrés. Les scripts `calibrate_qdm.py`, `calibrate_qdm_arome_native.py` et le
  pipeline statistique utilisent maintenant cette classe. Leurs noms de fichiers
  et le type d'artefact `qdm` sont conservés pour le contrat de chargement.
- `QuantileDeltaMapping` : QDM climatique, ajusté sur modèle et observations de
  référence, appliqué à **une fenêtre climatique simulée complète**. Pas de
  dépendance aux observations futures, mais une dépendance à la distribution du
  modèle dans cette fenêtre. Ne pas l'appliquer émission par émission.

Selon [Cannon et al. (2015)](https://doi.org/10.1175/JCLI-D-14-00754.1), au rang
`p = F_mod_target(x)`, le résultat additif est
`Q_obs_ref(p) + x - Q_mod_ref(p)` ; le résultat multiplicatif est
`Q_obs_ref(p) * x / Q_mod_ref(p)`. Voir aussi la
[formulation de xsdba](https://xsdba.readthedocs.io/en/latest/_modules/xsdba/adjustment.html#QuantileDeltaMapping).
Les tests utilisent un oracle analytique indépendant (distributions affines),
pas une comparaison numérique exécutée contre xsdba.

Les PPF de référence sont interpolées linéairement sur une grille incluant les
extrêmes. Le rang empirique futur est `(rang - 1)/(n - 1)` ; les ex æquo prennent
le rang moyen. Pour les groupes mensuels, xarray fournit le mois calendaire ;
les jours bissextiles restent dans leur mois. Les dimensions spatiales sont
mises en commun : sélectionner une station avant une correction locale.
Chaque groupe nécessite dix valeurs de référence et deux valeurs cibles finies.
Une référence manquante déclenche une erreur en climat. Les valeurs non finies
sont exclues des distributions et deviennent NaN en sortie.

Pour les précipitations, cette fondation préserve les ratios **conditionnels aux
jours humides**, au-dessus de `wet_threshold` dans les unités d'entrée. Les jours
secs cibles deviennent zéro ; leur fréquence n'est pas ajustée. Les références
entièrement sèches ne permettent pas d'estimer un ratio et sont refusées. Les
quantiles avec ex æquo ou une grille grossière ne garantissent pas une égalité
exacte des quantiles de sortie après réordonnancement.

Les anciens joblib `QuantileDeltaMapping`, sans marqueur de version climatique,
conservent le QM historique avec un avertissement. Un nouveau `fit` active le
QDM climatique. Pour recalibrer un produit météo, utiliser explicitement
`WeatherQuantileMapping`. Les scores du benchmark empirique AROME ne sont pas
transférables automatiquement à ces calibrateurs sérialisés.

## Prototype reproductible

```bash
python -m downscaling.counterfactual --seed 127 --output reports/counterfactual/synthetic.json
```

Le générateur combine cycle saisonnier, bruit journalier, variabilité
interannuelle, biais d'observation synthétique et changement de moyenne/variance.
Aucun téléchargement ni entraînement deep learning.

```python
from downscaling.counterfactual import analyze

result = analyze(
    modeled_reference, observed_reference, modeled_target,
    index="spring_tmin",  # ou frost_days / daytime_tmax
    baseline="climat station X, 1981-2000",
    reference="observations station X, version et chemin de données",
    run_ref="station-x-scenario-a-run-001",
    months=(3, 4, 5),
    block_years=2,
    n_bootstrap=1000,
    confidence=0.95,
    seed=127,
)
result.to_json("reports/counterfactual/station_x.json")
```

Entrées : `xarray.DataArray(time)` locales, dates grégoriennes journalières à
minuit, unités explicites `degC` ou `K`. Le modèle et les observations partagent
les dates de référence ; la période cible est disjointe. Les saisons doivent
être complètes, sans valeurs manquantes. Les années doivent être consécutives.

- `spring_tmin` : moyenne annuelle saisonnière des Tmin journalières.
- `frost_days` : nombre annuel saisonnier de jours avec Tmin strictement sous
  `frost_threshold` (0 °C par défaut).
- `daytime_tmax` : moyenne annuelle saisonnière des Tmax diurnes fournies ;
  l'appelant doit définir et appliquer la fenêtre diurne en amont.

Les trois métriques sont l'espérance sous climat de référence (observations),
l'espérance cible corrigée par QDM et leur différence cible moins référence.
Chaque année a le même poids. Le bootstrap indépendant par période tire des
années entières, ou des blocs mobiles non circulaires de plusieurs années,
puis construit des intervalles percentiles pour les trois estimations.

## Traçabilité et limites (#115)

Chaque entrée JSON porte `protocol`, `baseline`, `reference`, `run_ref`, référence
méthodologique, indice, unités, seuil, mois, années, effectifs et paramètres du
bootstrap/QDM. Le démonstrateur ajoute commande, graine, SHA Git et état modifié.
`run_ref` identifie l'exécution reproductible ; le CLI accepte `--run-ref` et
utilise le chemin de sortie par défaut. La persistance refuse une provenance
absente et les nombres non finis.

Le transfert est univarié et suppose une stationnarité de la correction de biais
entre référence et cible ; il ne préserve pas les dépendances multivariées.

Les intervalles sont **conditionnels à la calibration QDM fixée** : ils couvrent
l'échantillonnage interannuel, pas l'incertitude de calibration, des modèles,
des scénarios ou des capteurs. La dépendance entre périodes n'est pas estimée.
Préserver un signal simulé ne valide pas sa physique locale et ne constitue pas
une attribution causale aux émissions anthropiques. Ce prototype ne mesure pas
non plus une probabilité d'attribution d'événement.

Aucune donnée réelle n'est embarquée dans cet exemple. Les données AROME/Sencrop
référencées par les scripts météo nécessitent préparation journalière, contrôle
de complétude et une simulation couvrant deux périodes climatiques appropriées.

import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata

# Cargar datasets
titanic = pd.read_csv("/home/alex/demo/data-generators/titanic.csv")
resample = pd.read_csv("/home/alex/demo/data-generators/generated-data/titanic_resample.csv")
smote = pd.read_csv("/home/alex/demo/data-generators/generated-data/titanic_smote.csv")
gmm = pd.read_csv("/home/alex/demo/data-generators/generated-data/titanic_gmm.csv")
copula = pd.read_csv("/home/alex/demo/data-generators/generated-data/titanic_copula.csv")
ctgan = pd.read_csv("/home/alex/demo/data-generators/generated-data/titanic_ctgan.csv")
# Crear metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=titanic)

# Convertir a diccionario para sdmetrics
metadata_dict = metadata.to_dict()

# Generar reporte
my_report = QualityReport()
my_report.generate(real_data=titanic, synthetic_data=resample, metadata=metadata_dict)

print("✅ Overall Quality Score Resample:", my_report.get_score())

# Generar reporte
my_report = QualityReport()
my_report.generate(real_data=titanic, synthetic_data=smote, metadata=metadata_dict)

print("✅ Overall Quality Score Smote:", my_report.get_score())

# Generar reporte
my_report = QualityReport()
my_report.generate(real_data=titanic, synthetic_data=gmm, metadata=metadata_dict)

print("✅ Overall Quality Score GMM:", my_report.get_score())


# Generar reporte
my_report = QualityReport()
my_report.generate(real_data=titanic, synthetic_data=copula, metadata=metadata_dict)

print("✅ Overall Quality Score COPULA:", my_report.get_score())

my_report = QualityReport()
my_report.generate(real_data=titanic, synthetic_data=ctgan, metadata=metadata_dict)

print("✅ Overall Quality Score CTGAN:", my_report.get_score())
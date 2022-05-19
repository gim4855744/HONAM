import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_california_housing


def california_housing_prices():

    data_dir = "./Datasets/CaliforniaHousingPrices/"
    data_path = os.path.join(data_dir, "california_housing_prices.csv")
    data = pd.read_csv(data_path)

    data.dropna(axis=1, inplace=True)
    data.rename(columns={"median_house_value": "target"}, inplace=True)

    return data


def sklearn_california_housing():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
    data['target'] = california_housing.target
    return data


def house_prices():

    data_dir = "./Datasets/HousePrices/"
    data_path = os.path.join(data_dir, "house_prices.csv")
    data = pd.read_csv(data_path).drop("Id", axis=1)

    data.dropna(axis=1, inplace=True)
    data["MSSubClass"] = data["MSSubClass"].astype("object")

    data.rename(columns={"SalePrice": "target"}, inplace=True)

    return data


def insurance():

    data_dir = "./Datasets/Insurances/"
    data_path = os.path.join(data_dir, "insurance.csv")
    data = pd.read_csv(data_path)

    data.dropna(axis=1, inplace=True)
    data.rename(columns={"charges": "target"}, inplace=True)

    return data


def fico():

    data_dir = "./Datasets/Fico/"
    data_path = os.path.join(data_dir, "fico.csv")
    data = pd.read_csv(data_path)

    data.dropna(axis=1, inplace=True)
    data.rename(columns={"RiskPerformance": "target"}, inplace=True)

    data.target = LabelEncoder().fit_transform(data.target)

    return data


def credit():

    data_dir = "./Datasets/Credit/"
    data_path = os.path.join(data_dir, "credit.csv")
    data = pd.read_csv(data_path)

    drop_columns = ["Unnamed: 0", "Time"]
    data = data.drop(drop_columns, axis=1)

    data.dropna(axis=1, inplace=True)
    data.rename(columns={"Class": "target"}, inplace=True)

    data.target = LabelEncoder().fit_transform(data.target)

    return data


def mimic():

    data_dir = "./Datasets/mimic-iv/"

    admissions = pd.read_csv(os.path.join(data_dir, "core/admissions.csv"))
    patients = pd.read_csv(os.path.join(data_dir, "core/patients.csv"))

    admissions.loc[admissions.deathtime.isna(), "target"] = 0
    admissions.loc[admissions.deathtime.notna(), "target"] = 1

    admissions = admissions.drop(["admittime", "dischtime", "deathtime", "edregtime", "edouttime", "discharge_location", "hospital_expire_flag"], axis=1)
    patients = patients.drop(["anchor_year", "anchor_year_group", "dod"], axis=1)

    admissions = admissions.dropna()
    patients = patients.dropna()

    admissions_patients = pd.merge(admissions, patients, on="subject_id")
    admissions_patients = admissions_patients.drop(["subject_id", "hadm_id"], axis=1)

    # if using icd codes, uncomment this lines
    # diagnoses_icd = pd.read_csv(os.path.join(data_dir, "hosp/diagnoses_icd.csv"))
    # diagnoses_icd = diagnoses_icd.drop(["icd_version"], axis=1)
    # diagnoses_icd = diagnoses_icd.dropna()
    # icd_code = diagnoses_icd.groupby(by="hadm_id").agg({"icd_code": ','.join})
    # admissions_patients_icd = pd.merge(admissions_patients, icd_code, on="hadm_id")

    return admissions_patients


def compas():

    data_dir = "./Datasets/COMPAS/"
    data_path = os.path.join(data_dir, "compas-scores.csv")

    data = pd.read_csv(data_path)
    data = data.get(["sex", "age", "race", "priors_count", "score_text"])
    data = data.dropna(axis=0)

    condition = (data.score_text != "Low").astype("int")
    data["target"] = condition

    data = data.drop("score_text", axis=1)

    return data

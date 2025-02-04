CREATE TABLE MedicationAdministrationComponentFactX (
    MedicationAdministrationComponentKey INTEGER  PRIMARY KEY,
    PatientDurableKey                    INTEGER  REFERENCES PatientDim (DurableKey),
    ComponentMedicationKey               INTEGER  REFERENCES MedicationDim (MedicationKey),
    AdministrationInstant                DATETIME,
    AdministrationDurationMinutes        REAL,
    AdministrationActionGroup            TEXT,
    AdministrationRoute                  TEXT,
    ComponentType                        TEXT,
    ComponentMedicationLine              INTEGER,
    AdministrationDose                   REAL,
    AdministrationRate                   REAL,
    ComponentMedicationAmount            REAL,
    AdministrationDoseUnit               TEXT,
    AdministrationRateUnit               TEXT,
    ComponentMedicationAmountUnit        TEXT,
    AdministrationEncounterKey           INTEGER  REFERENCES EncounterFact (EncounterKey),
    MedicationOrderEpicId                INTEGER
);

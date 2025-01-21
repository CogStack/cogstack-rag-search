CREATE TABLE MedicationCodeMappingDimX (
    MedicationCodeMappingKey INTEGER PRIMARY KEY,
    MedicationKey            INTEGER REFERENCES MedicationDim (MedicationKey),
    Code                     TEXT
);

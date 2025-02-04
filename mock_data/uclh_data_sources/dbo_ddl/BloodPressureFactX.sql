CREATE TABLE BloodPressureFactX (
  BloodPressureKey      INTEGER  PRIMARY KEY,
  PatientDurableKey     INTEGER  REFERENCES PatientDim (DurableKey),
  Systolic              INTEGER,
  Diastolic             INTEGER,
  MeanArterialPressure  INTEGER,
  SystolicSnomed        TEXT,
  DiastolicSnomed       TEXT,
  MeanArterialPressureSnomed TEXT,
  TakenInstant          DATETIME,
  EncounterKey          INTEGER  REFERENCES EncounterFact (EncounterKey)
);

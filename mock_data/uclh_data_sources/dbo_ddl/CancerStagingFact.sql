CREATE TABLE CancerStagingFact (
    PatientDurableKey INTEGER REFERENCES PatientDim (DurableKey), 
    EncounterKey      INTEGER  REFERENCES EncounterFact (EncounterKey),
    CancerStagingKey   INTEGER PRIMARY KEY
);
  
CREATE TABLE DiagnosisTerminologyDim (
     DiagnosisTerminologyKey INTEGER PRIMARY KEY, 
     DiagnosisKey            INTEGER REFERENCES DiagnosisDim (DiagnosisKey), 
     Type                    TEXT, 
     Value                   TEXT,
     MappingLine_X           INTEGER
);
 
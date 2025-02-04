CREATE TABLE CancerStagingAttributeValueDim (
    CancerStagingKey   INTEGER REFERENCES CancerStagingFact (CancerStagingKey), 
    AttributeKey       INTEGER REFERENCES AttributeDim (AttributeKey),
    SmartDataElementEpicId TEXT,
    DateKey            INTEGER,
    StringValue        TEXT,
    NumericValue       FLOAT,
    DateValue          TEXT,
    AttributeType      TEXT
);
  
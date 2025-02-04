CREATE TABLE SocialFactorsX (
	SocialFactorsKey				      INTEGER  PRIMARY KEY,
	EncounterKey					        INTEGER  REFERENCES EncounterFact (EncounterKey),
	AlcoholBingeDrinking			    VARCHAR,
	AlcoholBingeDrinkingCode		  INTEGER,
	AlcoholDrinksPerDay				    VARCHAR,
	AlcoholDrinksPerDayCode			  INTEGER,
	AlcoholDrinkFrequency			    VARCHAR,
	AlcoholDrinkFrequencyCode		  INTEGER,
	AlcoholOuncesPerWeek			    VARCHAR,
	AlcoholInformationSource		  VARCHAR,
	AlcoholInformationSourceCode	INTEGER,
	AlcoholUser						        VARCHAR,
	AlcoholUserCode					      INTEGER,
	AlcoholComment 					      VARCHAR,
	HistoryContactDateKey         INTEGER REFERENCES DateDim (DateKey),
	TobaccoSmoking         	      VARCHAR
    );
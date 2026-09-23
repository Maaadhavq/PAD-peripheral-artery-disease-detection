# Feature dictionary

The 13 features the PAD model uses, where each comes from in MIMIC-IV, and what
it means clinically. This file is part of the copilot's knowledge base, so the
assistant can explain what a feature *is* and not only how much it moved a
prediction.

## gender

Binary, 1 for male and 0 otherwise, from `patients.gender`. PAD prevalence is
higher in men at a given age, though women are diagnosed later and more often
present without classic claudication.

## age at admission

Years, computed as `patients.anchor_age + (admission year - patients.anchor_year)`.
This is how MIMIC-IV encodes age while preserving date shifting. Age is the
strongest single demographic risk factor for PAD.

## total cholesterol

mg/dL, mean of all `Cholesterol, Total` results recorded during the index
admission (`labevents`). PAD is caused by atherosclerosis, and circulating
cholesterol drives plaque formation in peripheral arteries the same way it does
in coronary arteries.

## glucose

mg/dL, mean of all glucose results during the index admission. Several
`itemid`s map to the glucose label — point-of-care and laboratory assays — and
the pipeline averages them rather than keeping them as separate columns.
Elevated glucose acts as a proxy for diabetes and for glycaemic control.

## creatinine

mg/dL, mean during the index admission. A marker of kidney function. Chronic
kidney disease and PAD share risk factors and frequently co-occur, so
creatinine carries vascular information beyond the kidney itself.

## hemoglobin

g/dL, mean during the index admission. Anaemia reduces oxygen delivery to
already under-perfused limbs and is associated with worse outcomes in PAD.

## platelet count

K/uL, mean during the index admission. Platelets drive thrombus formation on
ruptured atherosclerotic plaque, which is the mechanism behind acute limb
ischaemia.

## diabetes history

Binary. Set when an ICD-9 `250*` or ICD-10 `E10/E11/E13*` code appears on an
admission *strictly before* the index admission. Diabetes is among the
strongest risk factors for PAD and for its progression to critical limb
ischaemia.

## hypertension history

Binary. ICD-9 `401-405*` or ICD-10 `I10-I13, I15*`, again from strictly prior
admissions. Sustained high pressure accelerates arterial damage and plaque
formation.

## heart disease history

Binary. ICD-9 `410-414*` or ICD-10 `I20-I25*`. Coronary artery disease and PAD
are the same disease process in different arterial beds, so one strongly
predicts the other.

## stroke history

Binary. ICD-9 `430-436*` or ICD-10 `I60-I64*`. Cerebrovascular disease is the
third major atherosclerotic territory alongside coronary and peripheral.

## statin therapy

Binary, set when a prior admission has a prescription matching an explicit
statin name (atorvastatin, rosuvastatin, simvastatin, pravastatin, lovastatin,
pitavastatin, fluvastatin). The pattern deliberately does not match the bare
substring "statin", because that also matches nystatin, an antifungal.

A statin flag is a marker of *treated* atherosclerotic risk: it usually means a
clinician had already judged the patient to be at cardiovascular risk.

## antiplatelet therapy

Binary, set from prior-admission prescriptions matching aspirin, clopidogrel,
Plavix, ticagrelor or prasugrel. Like statins, this is a marker that
cardiovascular risk had already been recognised and treated.

## Timing rules

Two different cutoffs apply, and the distinction matters when reading any
explanation:

- **Labs** are taken from the index admission itself. They are the measurements
  available when the patient presents.
- **Comorbidities and medications** come from *strictly earlier* admissions.
  ICD codes are assigned at discharge, so a code on the index admission is not
  known at admission time. Statins and antiplatelets are also the standard
  treatment for PAD, so counting prescriptions written during the index stay
  would feed the model a consequence of the diagnosis it is meant to predict.

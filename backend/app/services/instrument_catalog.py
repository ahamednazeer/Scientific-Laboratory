from sqlalchemy.orm import Session

from app.models import Instrument


DEFAULT_INSTRUMENTS = [
    {
        "name": "Microscope",
        "category": "optical",
        "description": "Used to magnify small specimens for detailed visual analysis.",
        "operation_steps": "1) Place slide. 2) Select objective lens. 3) Adjust focus knobs. 4) Tune illumination.",
        "safety_warnings": "Handle glass slides carefully. Avoid touching objective lenses directly.",
    },
    {
        "name": "Centrifuge",
        "category": "separation",
        "description": "Separates components by density using rotational force.",
        "operation_steps": "1) Balance tubes. 2) Set RPM/time. 3) Lock lid. 4) Start cycle. 5) Wait for complete stop.",
        "safety_warnings": "Never open while spinning. Ensure rotor balance to avoid mechanical failure.",
    },
    {
        "name": "Pipette",
        "category": "liquid-handling",
        "description": "Measures and transfers precise liquid volumes.",
        "operation_steps": "1) Attach tip. 2) Set volume. 3) Aspirate at first stop. 4) Dispense at second stop.",
        "safety_warnings": "Use correct tips. Avoid cross-contamination with frequent tip changes.",
    },
    {
        "name": "Bunsen Burner",
        "category": "heating",
        "description": "Provides controlled open flame for heating and sterilization.",
        "operation_steps": "1) Check tubing. 2) Ignite with striker. 3) Adjust air intake for blue flame.",
        "safety_warnings": "Keep flammables away. Tie back hair and wear eye protection.",
    },
    {
        "name": "Spectrophotometer",
        "category": "analysis",
        "description": "Measures absorbance/transmittance to quantify sample concentration.",
        "operation_steps": "1) Warm up device. 2) Blank with control cuvette. 3) Measure sample at target wavelength.",
        "safety_warnings": "Use clean cuvettes and avoid fingerprints on optical surfaces.",
    },
    {
        "name": "Hot Plate Stirrer",
        "category": "heating-mixing",
        "description": "Combines heating and magnetic stirring for uniform solution preparation.",
        "operation_steps": "1) Place vessel and stir bar. 2) Set temperature and speed gradually.",
        "safety_warnings": "Use heat-resistant gloves. Do not touch plate surface during/after use.",
    },
    {
        "name": "Analytical Balance",
        "category": "measurement",
        "description": "Measures mass with high precision for reagent preparation.",
        "operation_steps": "1) Tare container. 2) Add sample slowly. 3) Close draft shield before reading.",
        "safety_warnings": "Keep balance clean and vibration-free. Avoid corrosive spills.",
    },
    {
        "name": "pH Meter",
        "category": "analysis",
        "description": "Measures hydrogen ion activity to determine solution acidity/alkalinity.",
        "operation_steps": "1) Calibrate with buffers. 2) Rinse probe. 3) Immerse and wait for stable reading.",
        "safety_warnings": "Store electrode correctly and avoid scratching the glass bulb.",
    },
    {
        "name": "pH Strip",
        "category": "analysis",
        "description": "Colorimetric strip used for rapid acidity/alkalinity checks.",
        "operation_steps": "1) Dip strip briefly. 2) Remove excess liquid. 3) Compare color to chart immediately.",
        "safety_warnings": "Keep strips dry before use and avoid touching the reactive pad.",
    },
    {
        "name": "Test Tube Rack",
        "category": "support",
        "description": "Holds test tubes upright for safe organization during experiments.",
        "operation_steps": "1) Place rack on stable surface. 2) Insert labeled test tubes vertically.",
        "safety_warnings": "Avoid overcrowding and check for cracks in glass tubes.",
    },
    {
        "name": "Volumetric Flask",
        "category": "glassware",
        "description": "Prepares solutions with precise final volume.",
        "operation_steps": "1) Add solute. 2) Fill near mark. 3) Use dropper to meniscus line. 4) Cap and invert mix.",
        "safety_warnings": "Do not heat directly. Handle neck and stopper carefully.",
    },
    {
        "name": "Hot Plate",
        "category": "heating",
        "description": "Heats solutions and glassware without an open flame.",
        "operation_steps": "1) Place vessel centrally. 2) Increase temperature gradually. 3) Monitor continuously.",
        "safety_warnings": "Surface remains hot after use. Handle with heat-resistant gloves.",
    },
    {
        "name": "Spring Balance",
        "category": "measurement",
        "description": "Measures force or weight using spring extension.",
        "operation_steps": "1) Zero the scale. 2) Hang sample steadily. 3) Read value at eye level.",
        "safety_warnings": "Do not overload beyond rated capacity to prevent spring damage.",
    },
    {
        "name": "Laboratory Balance",
        "category": "measurement",
        "description": "General-purpose balance for routine mass measurements.",
        "operation_steps": "1) Level and tare balance. 2) Place sample container. 3) Record stable reading.",
        "safety_warnings": "Keep pan clean and avoid drafts or vibrations during measurements.",
    },
]


def seed_instruments_if_needed(db: Session) -> None:
    existing_names = {name for (name,) in db.query(Instrument.name).all()}
    added = False
    for payload in DEFAULT_INSTRUMENTS:
        if payload["name"] in existing_names:
            continue
        db.add(Instrument(**payload))
        added = True

    if added:
        db.commit()

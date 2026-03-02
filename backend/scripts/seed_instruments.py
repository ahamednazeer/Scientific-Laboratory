from app.db import Base, SessionLocal, engine
from app.services.instrument_catalog import seed_instruments_if_needed


if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        seed_instruments_if_needed(session)
        print("Instrument seed complete.")
    finally:
        session.close()

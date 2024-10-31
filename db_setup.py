from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Intent(Base):
    __tablename__ = 'intents'
    intent_id = Column(Integer, primary_key=True)
    intent_name = Column(String)

class Category(Base):
    __tablename__ = 'categories'
    category_id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('intents.intent_id'))
    category_name = Column(String)

class Question(Base):
    __tablename__ = 'questions'
    question_id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.category_id'))
    question_text = Column(String)

# Set up the database connection
engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def get_categories(intent_id, session):
    return session.query(Category).filter_by(intent_id=intent_id).all()

def get_questions(category_id, session):
    return session.query(Question).filter_by(category_id=category_id).all()

session = Session()

def populate_sample_data(session):
    if session.query(Intent).count() == 0:
        intent1 = Intent(intent_name='Technical Skill Assessments')
        intent2 = Intent(intent_name='Challenges')
        intent3 = Intent(intent_name='Solutions')

        session.add_all([intent1, intent2, intent3])
        session.commit()

        category1 = Category(intent_id=intent1.intent_id, category_name='Technical Challenges')
        category2 = Category(intent_id=intent2.intent_id, category_name='Team Collaboration Challenges')

        session.add_all([category1, category2])
        session.commit()

        question1 = Question(category_id=category1.category_id, question_text='What are some common technical challenges you face in your projects?')
        question2 = Question(category_id=category1.category_id, question_text='Can you describe a recent technical challenge and how you resolved it?')
        question3 = Question(category_id=category2.category_id, question_text='How do you handle conflicts within your team during a project?')
        question4 = Question(category_id=category2.category_id, question_text='What strategies do you use to enhance collaboration among team members?')

        # Add questions to the session
        session.add_all([question1, question2, question3, question4])
        session.commit()
        print("Sample data populated successfully.")
    else:
        print("Sample data already exists. Skipping population.")

# sample data
populate_sample_data(session)
session.close()

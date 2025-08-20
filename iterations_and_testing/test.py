import networkx as nx

# Import the system under test
from graph_of_thought import GraphOfThought, BiasDetector, BiasScores

# Placeholder functions for benchmark dataset loading
def load_stereoset_texts():
    """Load StereoSet-like texts with potential stereotypes"""
    return [
        "The doctor walked into the room. He was wearing a white coat and stethoscope.",
        "The nurse checked on the patient. She was very caring and gentle.",
        "The programmer worked late into the night. He debugged the complex algorithm.",
        "The teacher graded papers all evening. She prepared lessons for tomorrow.",
        "The CEO made important decisions. He led the company meeting confidently.",
        "The secretary organized the files. She scheduled all the appointments efficiently."
    ]

def load_biasbios_texts():
    """Load BiasBios-like professional biographies"""
    return [
        "John Smith is a successful software engineer at Google. He graduated from MIT and specializes in machine learning.",
        "Maria Garcia is a talented nurse at City Hospital. She has been caring for patients for over 10 years.",
        "David Chen is a skilled surgeon at Memorial Medical Center. He performs complex cardiac operations.",
        "Sarah Johnson is an experienced teacher at Lincoln Elementary. She has been educating children for 15 years.",
        "Michael Brown is a dedicated firefighter with the city department. He has saved countless lives.",
        "Lisa Wilson is a compassionate social worker. She helps families in crisis situations."
    ]

def load_hatecheck_texts():
    """Load HateCheck-like texts for hate speech detection"""
    return [
        "All immigrants are taking our jobs and should go back where they came from.",
        "People of different backgrounds contribute valuable perspectives to our community.",
        "Women are naturally worse at math and science than men.",
        "Everyone deserves equal opportunities regardless of their gender.",
        "Young people today are lazy and entitled, unlike previous generations.",
        "Each generation brings unique strengths and perspectives to society."
    ]

# Configuration

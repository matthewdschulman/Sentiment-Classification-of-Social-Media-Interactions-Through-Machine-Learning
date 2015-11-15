TRAINING DATASET FOR CLIPEval Task
_______________________________

__________________
Technical details
__________________
The sentences have been automatically extracted from a large corpus of English. 

__________________
Data descriptions
__________________
The training dataset contains 1280 sentences describing events reported in first person (singular or plural).
Each sentence is annotated with a polarity value (POSITIVE, NEGATIVE, NEUTRAL) and an event label which corresponds to the instantiations of psychologically grounded pleasant and unpleasant events macro-classes.

As the task aims at identifying polarity values of events, the annotation process has taken into account the notion of factivity. POSTIVE and NEGATIVE values have been attribute to factive events (i.e. events which actually occurred or happended). On the other hand, the value NEUTRAL has been attributed to non-factive events, including possibilities, and future events (es. "But no matter what, I will go bird watching at Mai Po again soon.")

A single sentence can be composed by multiple coordinated main clauses. When facing such a situation, there two possibile outcomes:

a.) each main clause is a mention of the same event instance or class (e.g. "I went to the swimming pool and later Dad brought me to the park" : two main clauses, same event instance - GOING_TO_PLACES)
b.) each main clause is a mention of a different event instance or class (e.g. "I went to the swimming pool and later I joined my girlfriend to the concert" : two main clauses, two event instances - GOING_TO_PLACES plus ATTENDING_EVENT)

regardless of both case a.) and b.) the assignment of the polarity value and of the event instance (class) has been perfoemed by taking into account only the first main clause. This means that in the example in a.), we considered for polarity and event instance assignment only the main clause "I went to the swimming pool" (event instance GOING_TO_PLACES). Similarly, for the example in b.), we considered only the first main clause for polarity and instance assignment, i.e "I went to the swimming pool" (event instance GOING_TO_PLACES).

There are 8 macro-instances or classes:
ATTENDING_EVENT
COMMUNICATION_ISSUE
GOING_TO_PLACES
LEGAL_ISSUE
MONEY_ISSUE
OUTDOOR_ACTIVITY
PERSONAL_CARE
(FEAR_OF)_PHYSICAL_PAIN


The input file for the task consists of three tab-separated fields:

SENTENCE <TAB> POLARITY_VALUE <TAB> events macro-class

-----------------
Task description
_________________
The CLIPEval task will be organized around two subtasks:

SUBTASK A: identify the polarity value associated to the event instance. Participants are required to associate each sentence with a polarity value (POSITIVE, NEGATIVE or NEUTRAL). Evaluation and ranking will be performed on F1 score for the polarity values. For this subtask, participants are not required to identify the event instantiation.

SUBTASK B: identify the event instantiations and associated polarity values. Participants are required to associate each sentence both with a class label of the event instance and the polarity value (POSITIVE, NEGATIVE or NEUTRAL). Training and test set will present a set of event instance labels. Evaluation and ranking will be done on F1 score on the correct identification of the event instance type and polarity value. 

______________
_________
Test data
_________
The test data will have the following format for both subtasks. It will consist of two tab-separated fields:

ID <TAB> SENTENCE <TAB> 

_____________
Output format
_____________
Output format for the evaluation must consist of three tab-separated fields for SUBTASK A:

ID <TAB> SENTENCE <TAB> POLARITY_VALUE


Output format for the evaluation of SUBTASK B must consist of four tab-separated fields:


ID <TAB> SENTENCE <TAB> POLARITY_VALUE <TAB> events macro-class
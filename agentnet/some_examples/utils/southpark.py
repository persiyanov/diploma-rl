"""
auxilary stuff used to parse southpark data
"""


from operator import add
is_EOC_marker = lambda (speaker,phrase): speaker in ['\n',None] or phrase in ['\n',None]


def parse_conversations(data):
    phrases = reduce(add,[eps['conversation']+[['\n','\n']] for eps in data])

    conversations=[]
    current_conversation = []
    for phrase in phrases:
        if is_EOC_marker(phrase):
            conversations.append(current_conversation)
            current_conversation = []
        else:
            current_conversation.append(phrase)

    if len(current_conversation) !=0:
        conversations.append(current_conversation)

    conversations = filter(len,conversations)

    return conversations

import json
def get_conversations(json_fname="sp.json"):
    with open(json_fname) as f:
        return parse_conversations(json.load(f))
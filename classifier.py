from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.5
df_data = pd.read_csv("eve_crash_classified - eve_crash_classified.csv")
df_topics = pd.read_csv("topics.csv")



class Topic(BaseModel):
    name: str = Field(..., description = "name of the topic")
    id: str = Field(..., description = "ID of the topic, sometimes uuid sometimes same as the topic name")
    definition: str= Field(..., description = "The definition of the topic")
    
class TopicLLM(BaseModel):
    name: str = Field(..., description = "name of the topic")
    id: str = Field(..., description = "ID of the topic, sometimes uuid sometimes same as the topic name")

class UtteranceClassification(BaseModel):
    """classification of a single utterance"""
    reasoning: str = Field(..., description='the reasoning steps')
    message_id : str = Field(..., description='the message id')
    topics : list[TopicLLM] = Field([], description="Zero or more topics identified. Leave empty if no appropriate topic found")
    review : str = Field(..., description="review the current classification and voice out ambuiguity")
    confidence_level : float = Field(..., description="confidence level of the answer between 0 and 1", ge=0.0, le=1.0)
    
    # or put the ' confidence level in {self_evaluation_clause} if inside

class ClassificationResponse(BaseModel):
    """The response structure of topic classification"""
    topic_identifications: list[UtteranceClassification] = Field([], description = 'idnetified topics')
    

from joblib.memory import Memory
memory = Memory(location = ".joblib")
@memory.cache( ignore = ["llm"])
def get_classification(llm: ChatGoogleGenerativeAI, messages, response_model):
    response = llm.with_structured_output(response_model, include_raw=True).invoke(messages)
    return response

def main():
    records = df_topics.to_dict(orient='records')
    topics_llm = [Topic.model_validate(r).model_dump() for r in records]
    topic_ids = set([i["id"] for i in topics_llm])
    topics_as_string = str(topics_llm)
    game_name = "Eve Online"
    game_intro_prompt = "EVE Online is a massive, player-driven space MMO set in a persistent single-shard universe. Players pilot customizable ships, mine resources, trade, build, explore, and engage in complex PvP warfare. The economy, politics, and warfare are fully player-run, with corporations and alliances controlling territory across high-security space, low-security space, and nullsec. Ship loss is permanent — destruction is part of gameplay — but players can always rebuild using in-game currency (ISK). Terms like crash, gate, hole, warp, TIDI, and nullbloc refer to in-game mechanics or social events, while crash out usually means disbanding, rage-quitting, or failing socially, not a technical crash. Technical crashes refer to the game client, servers, or API (ESI) failing."
    system_prompt_topics = (
        f"You are an expert in software and gaming. Find topics mentioned or discussed in each message. Remember that topics can be in a form of explicit mentions of items or can be indirect or implicit. One message can have several different topics."
        + (f"You are finding topics for chats in the game {game_name}. " if game_name else "")
        + (f"{game_intro_prompt}" if game_intro_prompt else "") + 
        f"Available topics:\n{topics_as_string}"
    )
    
    df_labeled = df_data.loc[:29]
    use_fields = ['message_id','content']
    x = df_labeled[use_fields].to_dict(orient = 'records')
    x_verify = {str(i["message_id"]): i for i in x}
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
    from itertools import islice
    
    i_max = 5
    i = 0
    verified = []
    need_refinement = []
    failed = []
    error_messages = []
    while (len(x_verify) > 0) and (i < i_max):
        x_verify_batch = list(islice(x_verify.items(), 0, 10))
        x_batch = {str(k):v for k,v in x_verify_batch}
        messages = [SystemMessage(system_prompt_topics),
                    HumanMessage(str(x_batch))] + error_messages
        classifications  = get_classification(llm, messages, ClassificationResponse)
        error_messages = []
        if classifications.get('parsing_error'):
            pass # retry
        else:
            parsed: ClassificationResponse = classifications.get('parsed')
            for c in parsed.topic_identifications:
                if c.message_id in x_batch and all(i.id in topic_ids for i in c.topics):
                    if c.confidence_level > HIGH_CONFIDENCE_THRESHOLD:
                        x_verify.pop(c.message_id)
                        verified.append(c)
                    elif c.confidence_level > LOW_CONFIDENCE_THRESHOLD:
                        need_refinement.append(c)
                    else:
                        # just retry
                        pass
                    
                    
                pass
            pass
        i += 1
    failed += x_verify
    verified_d = {i.message_id: i for i in verified}
    incorrect = [row for i, row in df_labeled.iterrows() if not bool(row.correct_classified)]
    for i in incorrect:
        i['reasoning'] = verified_d[str(i.message_id)].reasoning
        i['new_classification'] = [j.id for j in verified_d[str(i.message_id)].topics]
        i['new_correct'] = 'crashes' not in i['new_classification']
    new_correct = sum(i['new_correct']  for i in incorrect)
    print(f"false positive remove rate is {new_correct} out of {len(incorrect)}, "
          f"removal rate is {new_correct / len(incorrect):.2f}")
    df_corrected = pd.DataFrame(incorrect)
    df_corrected.to_csv('improved.csv')
if __name__ == "__main__":
    main()
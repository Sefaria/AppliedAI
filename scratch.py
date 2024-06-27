import os
import yaml
from langchain_core.documents import Document
from VirtualHavruta import VirtualHavruta
from VirtualHavruta.util import create_logger, part_res

# Replace with your paths
PATH_CONFIG = ""
PATH_PROMPTS = ""

with open(PATH_CONFIG) as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config["openai_model_api"]["api_key"]
log_path = config["files"]["log_path"]
logger = create_logger(f=log_path)

vh = VirtualHavruta(PATH_PROMPTS, PATH_CONFIG, logger)
query='Find references to the concept of ""Dying with your hand on your heart"  in Jewish literature'
msgid = "1234"

total_tokens = 0  # Initialize token counter

# Post an initial message acknowledging the received question
interim_msg = "I got your question. Let me get started to think."
# slack_post_msg(client, channel_id, message_ts, interim_msg, logger)

# Retrieve situational information, such as the current time and date
situ_info = vh.retrieve_situational_info(msgid)

# Handle possible adversarial attacks by checking the query content
detection, _, tok_count = vh.anti_attack(query, msgid)
total_tokens += tok_count  # Update the token count

# If detected, adapt the query, otherwise, edit the query
if "Y" in detection:
    screen_res, tok_count = vh.adaptor(query, msgid)
    total_tokens += tok_count

    if not screen_res or '@CANNOT-ADAPT@' in screen_res:
        interim_msg = "Umm... it seems there's little I could do for this question. Would you rephrase it and ask again?"
        # slack_post_msg(client, channel_id, message_ts, interim_msg, logger)
        # return
    
    # Inform the user that the question is being reconsidered
    interim_msg = f"Umm...I'm reconsidering your question and for now I interpret it as: {screen_res}"
    # slack_post_msg(client, channel_id, message_ts, interim_msg, logger)
else:
    screen_res, tok_count = vh.editor(query, msgid)
    total_tokens += tok_count

# Optimize the query for improved retrieval and response quality
translation, extraction, elaboration, quotation, challenge, proposal, tok_count = vh.optimizer(screen_res, msgid)
total_tokens += tok_count

if quotation:
    scripture_query = f"{part_res(quotation)} {part_res(extraction)}"
else:
    scripture_query = f"{part_res(translation)} {part_res(extraction)} {part_res(elaboration)}"
enriched_query = f"{part_res(translation)} {part_res(extraction)} {part_res(elaboration)} {part_res(proposal)} {part_res(quotation)}"


# Retrieve relevant documents and sort references for primary sources
p_retrieval_res: list[Document] = vh.retrieve_docs(scripture_query, msgid, 'primary')
# Retrieve relevant secondary documents from linker api
page_content_total: list[Document] = vh.retrieve_docs_linker(screen_res, enriched_query, msgid, 'primary')

# ----------------- New code starting here -----------------
# Exemplary use of the graph traversal retriever in the pipeline
retrieval_res_kg, token_count = vh.graph_traversal_retriever(screen_res=screen_res,
                                                          scripture_query=scripture_query,
                                                          enriched_query=enriched_query,
                                                          linker_results=page_content_total,
                                                          semantic_search_results=p_retrieval_res,  
                                                          filter_mode_nodes="primary",
                                                          )

sorted_src_rel_dict, src_data_dict, src_ref_dict = vh.merge_references_by_url(retrieval_res_kg)
sorted_src_rel_dict, src_data_dict, src_ref_dict = vh.merge_linker_refs(retrieved_docs=page_content_total,
                                                                        p_sorted_src_rel_dict=sorted_src_rel_dict,
                                                                        p_src_data_dict=src_data_dict,
                                                                        p_src_ref_dict=src_ref_dict,
                                                                        msg_id=msgid,
                                                                        )
ref_str = vh.generate_ref_str(sorted_src_rel_dict, src_data_dict, src_ref_dict, msg_id=msgid)
# ---- continue with vh.qa ------


import os

import yaml

from VirtualHavruta import VirtualHavruta
from VirtualHavruta.util import create_logger, part_res
# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"

PATH_CONFIG = "/Users/johannesgunterbirk/Documents/havruta/config.yaml"
PATH_PROMPTS = "/Users/johannesgunterbirk/Documents/havruta/prompts.yaml"

with open(PATH_CONFIG) as file:
    config = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = config["openai_model_api"]["api_key"]
log_path = config["files"]["log_path"]
logger = create_logger(f=log_path)

vh = VirtualHavruta(PATH_PROMPTS, PATH_CONFIG, logger)



query="What is in the mouth of a fool?"
msgid = "1234"

# start try clause slack app
try:
    total_tokens = 0  # Initialize token counter

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
    p_retrieval_res = vh.retrieve_docs(scripture_query, msgid, 'primary')
    # Retrieve relevant secondary documents from linker api
    page_content_total = vh.retrieve_docs_linker(screen_res ,enriched_query , msgid, 'secondary')
    # Sort primary references from KG vector database
    p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, tok_count = vh.sort_reference(enriched_query, p_retrieval_res, msgid, 'primary')
    
    # Merge secondary linker api results with original p_sorted_src_rel_dict, p_src_data_dict and p_src_ref_dict primary
    p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict = vh.merge_linker_refs(page_content_total, p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, msgid)
    
    # Retrieve relevant primary documents from linker api
    page_content_total = vh.retrieve_docs_linker(screen_res ,enriched_query , msgid, 'primary')
    
        # Merge primary linker api results with original p_sorted_src_rel_dict, p_src_data_dict and p_src_ref_dict primary
    p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict = vh.merge_linker_refs(page_content_total, p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, msgid)
    
    total_tokens += tok_count  # Update token count
    
    # Generate reference strings for primary sources
    p_conc_ref_data, p_citations, _, ref_count = vh.generate_ref_str(p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, msgid, 'primary')

    # Retrieve relevant documents and sort references for secondary sources
    s_retrieval_res = vh.retrieve_docs(enriched_query, msgid, 'secondary')
    
    # Sort secondary references from KG vector database
    s_sorted_src_rel_dict, s_src_data_dict, s_src_ref_dict, tok_count = vh.sort_reference(enriched_query, s_retrieval_res, msgid, 'secondary')
    # Merge secondary linker api results with original s_sorted_src_rel_dict, s_src_data_dict and s_src_ref_dict
    #s_sorted_src_rel_dict, s_src_data_dict, s_src_ref_dict = vh.merge_linker_refs(page_content_total, s_sorted_src_rel_dict, s_src_data_dict, s_src_ref_dict, msgid)
        
    total_tokens += tok_count  # Update token count

    # Generate reference strings and deeplinks for secondary sources
    s_conc_ref_data, s_citations, deeplinks, _ = vh.generate_ref_str(s_sorted_src_rel_dict, s_src_data_dict, s_src_ref_dict, msgid, 'secondary', ref_count)
    
    
    # Retrieve topic ontology results
    topic_ont_dict = vh.topic_ontology(extraction, msgid)
    
    # Check if the dictionary is empty and log a message
    if not topic_ont_dict:
        logger.info(f"MsgID={msgid}. No topics found for the given extraction.")
    else:
        topic_ont_results_msg = ""
        # Iterate over the result dictionary and add as a string to post at the end
        for topic, description in topic_ont_dict.items():
            topic_ont_results_msg += f"\n *Topic: {topic} *\nDescription: {description}\n"    

    # Combine primary and secondary reference data
    conc_ref_data = p_conc_ref_data + s_conc_ref_data + f"Ad-hoc supplementary information: {situ_info} {translation} {elaboration} {challenge} {proposal} {topic_ont_dict}"
    citations = p_citations + s_citations
    
    # Generate a deeplink to the Knowledge Graph if available
    neo4j_deeplink = vh.generate_kg_deeplink(deeplinks, msgid)

    # Formulate the main response based on the combined reference data
    main_response, tok_count = vh.qa(screen_res, conc_ref_data, msgid)
    total_tokens += tok_count  # Update token count

        
    if citations:
        if "@IRRELEVANT-SOURCE@" in main_response:
            final_msg = (
                "*I did find some references in my database, though they don't seem to strictly address the question per se. I'm trying to explain as below. In the meantime, it would be super helpful if you can provide more contexts and rephrase your question and then ask again.* \n\n"
                f"{part_res(main_response, '@IRRELEVANT-SOURCE@')}"
            )
            logger.warning(f"MsgID={msgid}. Model found irrelevant references!")
        else:
            final_msg = main_response + "\n"
        
        ref_msg = "\n *References:* \n" + citations + "\n"

        # If there are deeplinks available, add Knowledge Graph visualization message
        if deeplinks:
            kg_msg = (
                f"\n <{neo4j_deeplink}|*Visualize References in Knowledge Graph*> \n"
                "Discover the web of connections surrounding Sefaria with our "
                "Interactive Knowledge Graph Viewer built into Slack. \n"
            )
    else:
        # If no citations are available, inform the user
        final_msg = (
            "At the moment I couldn't find directly relevant information in my database. I appreciate your understanding."
            ""
        )
        ref_msg = "\n *References:* \n" + "No relevant references" + "\n"
        kg_msg = "\n *Visualize References in Knowledge Graph:* \n No knowledge graph to display. \n"
        
        
    # Log the total number of tokens spent processing this query
    logger.info(f"MsgID={msgid}. [Token Count] Spent {total_tokens} tokens in total for this round of query.")
except Exception as e:
    logger.error(f"MsgID={msgid}. [Pipeline Error] {e}")
# end try clause slack app

    
    # Log the total number of tokens spent processing this query
# node_id = vh.query_node_by_url(url="https://www.sefaria.org/Proverbs.14.3")
# screen_res, tok_count = vh.editor(query)
# translation, extraction, elaboration, quotation, challenge, proposal, tok_count = vh.optimizer(screen_res)
# enriched_query = f"{part_res(translation)} {part_res(extraction)} {part_res(elaboration)} {part_res(proposal)} {part_res(quotation)}"
# linker_res = vh.retrieve_docs_linker(screen_res=screen_res, enriched_query=enriched_query)

# docs_classic = vh.retrieve_docs(query=query)
# retrieval_res = vh.retrieve_top_1_doc_with_neighbors(query=query)    

# sorted_src_rel_dict, src_data_dict, src_ref_dict, total_tokens = vh.sort_reference(query=query, retrieval_res=retrieval_res, )

print("Done")
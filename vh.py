# Load basic libraries
import yaml, json
import operator
import pandas as pd
from datetime import datetime

# Import custom langchain modules for NLP operations and vector search
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.callbacks import get_openai_callback


# Main Virtual Havruta functionalities
class VirtualHavruta:
    def __init__(self, prompts_file: str, config_file: str, logger):
        with open(prompts_file, 'r') as f:
            self.prompts = yaml.safe_load(f)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize Neo4j vector index and retrieve DB configs
        model_api = self.config['openai_model_api']
        db = self.config['database']
        self.neo4j_vector = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(model=model_api['embedding_model']),
            index_name="index",
            url=db['db_url'],
            username=db['db_username'],
            password=db['db_password'],
        )
        self.top_k = db['top_k']
        self.neo4j_deeplink = db['neo4j_deeplink']

        # Initiate logger and pagerank lookup table
        self.logger = logger
        self.pr_table = pd.read_csv(self.config['files']['pr_table_path'])

        # Retrieve reference configs
        refs = self.config['references']
        self.primary_source_filter = refs['primary_source_filter']
        self.num_primary_citations = refs['num_primary_citations']
        self.num_secondary_citations = refs['num_secondary_citations']        

        # Initialize prompt templates and LLM instances
        self.initialize_prompt_templates()
        self.initialize_llm_instances()

    def initialize_prompt_templates(self):
        categories = ['anti_attack', 'adaptor', 'editor', 'optimization', 'classification']
        prompts = {'prompt_'+cat: self.create_prompt_template('system', cat) for cat in categories}
        self.prompt_qa = self.create_prompt_template('system', 'qa', True)
        self.__dict__.update(prompts)

    def create_prompt_template(self, category: str, template: str, ref_mode: bool = False) -> ChatPromptTemplate:
        system_message = SystemMessage(content=self.prompts[category][template])
        human_template = f"Question: {{human_input}}{' Reference Data: {ref_data}' if ref_mode else ''}."
        return ChatPromptTemplate.from_messages([
            system_message,
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def initialize_llm_instances(self):
        model_api = self.config['openai_model_api']
        chain_setups = self.config['llm_chain_setups']
        
        # Adding a condition to include json kwargs for models ending with '_json'
        for model_name, suffixes in chain_setups.items():
            model_kwargs = {"response_format": {"type": "json_object"}} if model_name.endswith('_json') else {}
            model_key = model_name.replace('_json', '')  # Removes the '_json' suffix for lookup in model_api
            setattr(self, model_name, ChatOpenAI(
                temperature=model_api.get(f"{model_key}_temperature", None),
                model=model_api.get(model_key, None),
                model_kwargs=model_kwargs
            ))
            self.initialize_llm_chains(getattr(self, model_name), suffixes)

    def initialize_llm_chains(self, model, suffixes):
        for suffix in suffixes:
            setattr(self, f"chat_llm_chain_{suffix}",
                    self.create_llm_chain(model, getattr(self, f"prompt_{suffix}")))

    def create_llm_chain(self, llm, prompt_template):
        return LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    def make_prediction(self, chain, query: str, action: str, msg_id: str='', ref_data: str = ''):
        with get_openai_callback() as cb:
            try: 
                res = chain.predict(human_input=query, ref_data=ref_data) if ref_data else chain.predict(human_input=query)
                self.logger.info(f"MsgID={msg_id}. Spent {cb.total_tokens} tokens for {action}. Result is {res}.")
            except Exception as e:
                self.logger.error(f"MsgID={msg_id}. Spent {cb.total_tokens} tokens for {action} but failed. Error is {e}.")
                res = ''
            return res, cb.total_tokens

    def anti_attack(self, query: str, msg_id: str=''):
        adv_res, tok_count = self.make_prediction(self.chat_llm_chain_anti_attack, query, "ANTI-ATTACK", msg_id)
        try:
            detection, explanation = adv_res.split('@SEP@')
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. Error occurred during attack detection: {e}.")
            detection, explanation = 'N', ''
        return detection, explanation, tok_count

    def adaptor(self, query: str, msg_id: str=''):
        adp_res, tok_count = self.make_prediction(self.chat_llm_chain_adaptor, query, "ADAPTATION", msg_id)
        return adp_res, tok_count

    def editor(self, query: str, msg_id: str=''):
        edit_res, tok_count = self.make_prediction(self.chat_llm_chain_editor, query, "EDITING", msg_id)
        return edit_res, tok_count
        
    def optimizer(self, query: str, msg_id: str=''):
        opt_res, tok_count = self.make_prediction(self.chat_llm_chain_optimization, query, "PROMPT OPTIMIZATION", msg_id)
        try:
            opt_res_json = json.loads(opt_res)
            translation = opt_res_json['Translation']
            extraction = opt_res_json['Key-Concepts']
            elaboration = opt_res_json['Elaboration']
            quotation = opt_res_json['Quotation']
            challenge = opt_res_json['Challenge']
            proposal = opt_res_json['Potential-Directions']
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. Error occurred during PROMPT OPTIMIZATION: {e}.")
            translation = extraction = elaboration = quotation = challenge = proposal = ''
        return translation, extraction, elaboration, quotation, challenge, proposal, tok_count

    def retrieve_docs(self, query: str, msg_id: str='', filter_mode: str='primary'):
        self.logger.info(f"MsgID={msg_id}. [RETRIEVAL] Retrieving {filter_mode} references using this query: {query}")
        # Convert primary_source_filter to a set for efficient lookup
        retrieved_docs = self.neo4j_vector.similarity_search_with_relevance_scores(
            query.lower(), self.top_k
        )
        # Filter the documents based on whether we're looking for primary or secondary sources
        if filter_mode == 'primary':
            predicate = lambda doc: any(s in doc[0].metadata['source'] for s in self.primary_source_filter)
        elif filter_mode == 'secondary':
            predicate = lambda doc: not any(s in doc[0].metadata['source'] for s in self.primary_source_filter)
        else:
            raise ValueError(f"MsgID={msg_id}. Invalid filter_mode: {filter_mode}")
        retrieval_res = list(filter(predicate, retrieved_docs))
        return retrieval_res
    
    def sort_reference(self, query: str, retrieval_res, msg_id: str = '', filter_mode: str='primary'):
        # Initialize dictionaries and tok_count to store the data, references, relevance scores, and token count
        src_data_dict = {}
        src_ref_dict = {}
        src_rel_dict = {}
        total_tokens = 0
        
        # Iterate over each item in the retrieval results
        for n, (d, sim_score) in enumerate(retrieval_res):
            # Concatenate reference data and its source
            ref_data = d.page_content + "... --Origin of this " + d.metadata["source"]
            # Log the reference data and its link
            self.logger.info(f"MsgID={msg_id}. \n{n+1} RefData={ref_data} \nLink is {d.metadata['URL']}. Similarity Score is {sim_score}")
            # Classify the reference data and get the token count
            ref_class, tok_count = self.classification(query, ref_data, msg_id)
            total_tokens += tok_count
    
            # If the reference class is not None or 0 (i.e., it is relevant)
            if ref_class:
                if filter_mode == 'primary':
                    # Retrieve the PageRank score for the URL of the current reference
                    pagerank = self.retrieve_pr_score(d.metadata["URL"], msg_id)
                    # Calculate the relevance score based on classification, similarity, and PageRank
                    rel_score = ref_class * sim_score * pagerank
                else:
                    rel_score = ref_class * sim_score
    
                # If the URL is not already in src_data_dict, add all reference information
                if d.metadata["URL"] not in src_data_dict:
                    src_data_dict[d.metadata["URL"]] = d.page_content
                    src_ref_dict[d.metadata["URL"]] = d.metadata["source"]
                    src_rel_dict[d.metadata["URL"]] = rel_score
                else:
                    # If the URL is already present, handle different versions or sources with the same URL
                    existing_content = src_data_dict[d.metadata["URL"]]
                    # Concatenate page content for the same URL
                    src_data_dict[d.metadata["URL"]] = "...".join([existing_content, d.page_content])
    
                    # Avoid duplicate source listings by separating with a pipe "|"
                    existing_ref = src_ref_dict[d.metadata["URL"]]
                    existing_ref_list = existing_ref.split(" | ")
                    if d.metadata["source"] not in existing_ref_list:
                        src_ref_dict[d.metadata["URL"]] = " | ".join([existing_ref, d.metadata["source"]])
    
                    # Update the relevance score with the maximum score between existing and new
                    existing_rel_score = src_rel_dict[d.metadata["URL"]]
                    src_rel_dict[d.metadata["URL"]] = max(existing_rel_score, rel_score)
    
        # Sort the source relevance dictionary based on scores in descending order
        sorted_src_rel_dict = dict(
            sorted(src_rel_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
    
        # Return the sorted source relevance dictionary, source data dictionary, source reference dictionary, and token count
        return sorted_src_rel_dict, src_data_dict, src_ref_dict, total_tokens
    
    def classification(self, query: str, ref_data: str, msg_id: str=''):
        # Classifiy the data with LLM
        ref_class, tok_count = self.make_prediction(
                    self.chat_llm_chain_classification, query, "CLASSIFICATION", msg_id, ref_data)
        try:
            ref_class = int(ref_class)
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. LLM CLASSIFICATION result was set to 0. Error message is {e}.")
            ref_class = 0
        return ref_class, tok_count

    def retrieve_pr_score(self, doc_id: str, msg_id: str=''):
        page_ranks = self.pr_table.loc[self.pr_table['metadata.url'] == doc_id, 'metadata.pagerank']
        if not page_ranks.empty:
            pr_score = page_ranks.max()
            self.logger.info(f"MsgID={msg_id}. Retrieved pagerank score={pr_score}. Link is {doc_id}.")
        else:
            pr_score = 0
            self.logger.warning("MsgID={msg_id}. Cannot retrieve pagerank score. Link is {doc_id}.")
        return pr_score

    def generate_ref_str(self, sorted_src_rel_dict, src_data_dict, src_ref_dict, msg_id: str = '', ref_mode: str = 'primary', n_citation_base: int = 0) -> str:
        # Determine the starting citation number and how many citations to include
        n_citation_base = 0 if ref_mode == 'primary' else n_citation_base
        num_citations = self.num_primary_citations if ref_mode == 'primary' else self.num_secondary_citations

        # Lists to hold parts of the final strings
        ref_data_parts = []
        citation_parts = []
        deeplinks = []
        n_citation = 0

        # Process only the needed citations
        for n, (k, rel_score) in enumerate(sorted_src_rel_dict.items()):
            if n >= num_citations:
                break

            n_citation = n_citation_base + n + 1
            ref_data_parts.append(f"\n #Reference {n_citation}# {src_data_dict[k]}... --Origin of this {src_ref_dict[k]} \n")
            citation_parts.append(f"\n{n_citation}. {src_ref_dict[k]}")

            # Additional actions for secondary references
            if ref_mode == 'secondary':
                deeplinks.append(k)

            self.logger.info(
                    f"MsgID={msg_id}. Included this reference for {ref_mode} references: {k}   "
                    f"Relevance score = {sorted_src_rel_dict[k]}."
                )

        # Join the parts into final strings
        conc_ref_data = ''.join(ref_data_parts)
        citations = ''.join(citation_parts)

        return conc_ref_data, citations, deeplinks, n_citation

    def generate_kg_deeplink(self, deeplinks, msg_id: str=''):
        '''
        Currently KG visualization dashboard only displays up to the first three secondary references.
        '''
        # Define the maximum number of deeplinks to be used
        max_deeplinks_count = 3

        if deeplinks:
            idx = []
            for i in range(max_deeplinks_count):
                if i < len(deeplinks):
                    idx.append(i)
                else:
                    idx.append(-1)
            
            # TODO This is the current setting of the Neo4J dashaboard. Should reconsider in the future.
            neo4j_deeplink = (
                self.neo4j_deeplink
                + deeplinks[idx[0]]
                + "&neodash_url2="
                + deeplinks[idx[1]]
                + "&neodash_url3="
                + deeplinks[idx[2]]
            )
            self.logger.info(f"MsgID={msg_id}. Created KG deep link for secondary references: {neo4j_deeplink}.")
        else:
            neo4j_deeplink = ""
            self.logger.info(f"MsgID={msg_id}. Empty KG deep link for secondary references.")
        return neo4j_deeplink

    def qa(self, query: str, ref_data: str, msg_id: str=''):
        response, tok_count = self.make_prediction(
                    self.chat_llm_chain_qa, query, "qa", msg_id, ref_data)
        return response, tok_count

    def retrieve_situational_info(self, msg_id: str = ''):
        now = datetime.now()
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = now.weekday() 
        situ_info = f"\n [Situational Info] The time of asking this question is {days[day_of_week]}, {now.strftime('%d/%m/%Y %H:%M:%S')} \n"
        self.logger.info(f"MsgID={msg_id}. SITUATIONAL INFO: {situ_info}")
        return situ_info

    def query_sefaria_linker(self, text_title="", text_body="", with_text=1, debug=0, max_segments=5, msg_id: str = ''):
        # Sefaria Linker API endpoint
        api_url = "https://www.sefaria.org/api/find-refs"

        # Assemble headers and data for the POST request
        headers = {'Content-Type': 'application/json'}
        
        # Assemble the body of the POST request using a dictionary and directly pass it to requests.post
        data = {
            "text": {
                "title": text_title,
                "body": text_body,
            }
        }
        
         # Consolidate parameters, including those passed to the function
        params = {'with_text': with_text, 'debug': debug, 'max_segments': max_segments}

        try:
            # Simplify request by directly passing a dictionary to json parameter, which requests will automatically serialize
            response = requests.post(api_url, headers=headers, params=params, json=data)
            self.logger.info(f"MsgID={msg_id}. Sefaria linker query response: {response}. {response.json()}.")
            response.raise_for_status()  # Handles HTTP errors by raising an HTTPError exception for bad requests
            # response.json() will return the JSON response for a successful request
            return response.json()
        except requests.HTTPError as http_err:
            self.logger.error(f"MsgID={msg_id}. HTTP error occurred: {http_err}.") # Specific HTTP error handling
            return f"HTTP error occurred: {http_err}"  
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. Error occurred during Sefaria Linker Querying: {e}.") # General error handling
            return f"Error occurred during Sefaria Linker Querying: {e}"

    def retrieve_docs_linker(self, screen_res: str, enriched_query: str, msg_id: str = '', filter_mode: str = 'primary'):
        # Making a call to sefaria linker api
        json_input = self.query_sefaria_linker(text_title=screen_res, text_body=enriched_query)

        # To store documents
        results = []

        # Define predicate functions based on filter_mode
        if filter_mode == 'primary':
            predicate = lambda category: category in self.linker_primary_source_filter
        elif filter_mode == 'secondary':
            predicate = lambda category: category not in self.linker_primary_source_filter
        else:
            raise ValueError(f"Invalid filter_mode: {msg_id} - {filter_mode}")

        # Recursive function to traverse and collect data
        def traverse(json_data):
            if isinstance(json_data, dict):
                for key, value in json_data.items():
                    if key == 'refData':
                        # Process its children if it's 'refData'
                        for sub_key, sub_value in value.items():
                            # Apply the filtering predicate on the 'primaryCategory' field
                            if 'primaryCategory' in sub_value and predicate(sub_value['primaryCategory']):
                                # Add the page_rank to each document
                                # PR score is initialized to 6.0 for Sefaria Linker API
                                sub_value['page_rank'] = 6.0
                                results.append(sub_value)
                    elif isinstance(value, (dict, list)):
                        # Continue search in deeper levels
                        traverse(value)
            elif isinstance(json_data, list):
                for item in json_data:
                    traverse(item)

        traverse(json_input)
        self.logger.info(f"MsgID={msg_id}. Sefaria linker document retrieval results: {results}")
        return results

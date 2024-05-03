# Load basic libraries
import yaml, json
import operator
import pandas as pd
from datetime import datetime

# Import custom langchain modules for NLP operations and vector search
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks import get_openai_callback
import traceback
import requests

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
        linker_references = self.config['linker_references']
        self.primary_source_filter = refs['primary_source_filter']
        self.num_primary_citations = refs['num_primary_citations']
        self.num_secondary_citations = refs['num_secondary_citations']        
        self.linker_primary_source_filter = linker_references['primary_source_filter']
        
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
        '''
        Retrieves documents that match a specified query and filters them based on whether they are primary or secondary sources, using a similarity search.

        This function performs a similarity search based on the provided query and retrieves documents that either match the characteristics of primary or secondary sources as defined by a filter set. The results are filtered by checking each document's metadata against a predefined set of source filters. The function logs the process to ensure transparency and is equipped to handle errors related to invalid filter modes, raising a ValueError if necessary.
        
        Parameters:
        query (str): The query string used to search for relevant documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        filter_mode (str): The mode to filter the search results by 'primary' or 'secondary' to determine the relevance of the sources.
        
        Returns:
        list: A list of documents that meet the criteria of the specified filter mode, either as primary or secondary sources.
        
        Raises:
        ValueError: If an invalid filter_mode is provided, an exception is raised to indicate the error.
        '''
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
        '''
        Sorts and processes retrieval results for references based on their relevance to a given query, considering both primary and secondary filtering modes.
        
        This function processes a set of retrieval results, classifying each result for relevance and calculating a composite relevance score based on classification results, similarity scores, and, for primary references, PageRank scores. It also consolidates results with the same URL to avoid duplication, ensuring that the most relevant and comprehensive content is retained. The function logs each step for transparency and debugging purposes and returns dictionaries containing sorted relevance data, source data, and reference details, along with the total count of tokens used in processing.
        
        Parameters:
        query (str): The query string against which references are being sorted and classified.
        retrieval_res (iterable): An iterable of tuples containing reference data objects and similarity scores.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        filter_mode (str): Determines the mode for filtering references; can be 'primary' or 'secondary'. This affects how relevance scores are calculated.
        
        Returns:
        tuple: A tuple containing sorted source relevance dictionary, source data dictionary, source reference dictionary, and the total token count used during the process.
        
        Notes:
        The function is robust to variations in data and manages complex scenarios where multiple references may have the same URL but different content or sources. It effectively manages and logs all operations to ensure data integrity and traceability.
        '''
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
        '''
        Classifies the provided query and reference data using a chained language model, returning the classification result and token count.

        This function sends a query and reference data to a language model specifically tuned for classification tasks. It captures the classification result, which is expected to be a numerical value, and the count of tokens used by the model. If the model's output cannot be converted to an integer due to an error, the function logs the error and defaults the classification to 0. This ensures robust error handling and maintains the integrity of the classification process under all conditions.
        
        Parameters:
        query (str): The query string to be classified by the model.
        ref_data (str): Reference data related to the query that may influence the classification.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the classification result (int) and the token count (int) used in generating that result.
        
        Raises:
        Exception: Catches and logs any exception that occurs during the classification conversion process, defaulting the result to 0.
        '''
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
        '''
        Retrieves the page rank score for a given document identifier from a pre-defined page rank table.
        
        This function searches a dataframe for the page rank score associated with a specific document identifier (URL). If found, it returns the highest score present for that identifier; if no data is available, it returns zero and logs a warning. This function is critical for evaluating the importance or relevance of documents based on their page rank in various processing and decision-making contexts.
        
        Parameters:
        doc_id (str): The document identifier for which the page rank score is to be retrieved.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        int or float: The highest page rank score found for the given document identifier. Returns 0 if no score is found.
        
        Notes:
        The function uses logging to provide transparency about the retrieval process and to document any issues encountered, such as missing data for the specified document identifier.
        '''
        page_ranks = self.pr_table.loc[self.pr_table['metadata.url'] == doc_id, 'metadata.pagerank']
        if not page_ranks.empty:
            pr_score = page_ranks.max()
            self.logger.info(f"MsgID={msg_id}. Retrieved pagerank score={pr_score}. Link is {doc_id}.")
        else:
            pr_score = 0
            self.logger.warning("MsgID={msg_id}. Cannot retrieve pagerank score. Link is {doc_id}.")
        return pr_score

    def generate_ref_str(self, sorted_src_rel_dict, src_data_dict, src_ref_dict, msg_id: str = '', ref_mode: str = 'primary', n_citation_base: int = 0) -> str:
        '''
        Constructs formatted reference strings and citation lists based on the source relevance and data dictionaries, with specific handling for primary and secondary references.
        
        This function dynamically generates reference content and citation indices from sorted relevance data, differentiated by primary and secondary modes. It assembles detailed reference strings, citation lists, and deep links for secondary references if applicable. The function allows for continuation of citation numbering from a specified base, which is useful in documents where references span multiple sections or components.
        
        Parameters:
        sorted_src_rel_dict (dict): A dictionary sorted by relevance, mapping source identifiers to their relevance scores.
        src_data_dict (dict): A dictionary mapping source identifiers to their data content.
        src_ref_dict (dict): A dictionary mapping source identifiers to their reference details.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        ref_mode (str): Mode of reference; can be 'primary' or 'secondary'. This affects the starting citation number and how citations are processed.
        n_citation_base (int): The starting index for citations, which is only used in 'secondary' mode.
        
        Returns:
        tuple: A tuple containing strings for the concatenated reference data and citations, a list of deep links for secondary references, and the final citation index used.
        
        Notes:
        The function logs details of the reference processing for tracking and debugging, enhancing traceability and accountability in system operations involving reference data management.
        '''
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
        Generates a Knowledge Graph (KG) deep link URL by concatenating up to the first three secondary reference URLs provided.
        
        This function constructs a URL for the Neo4J dashboard by using up to three deep links from the provided list. If there are fewer than three deep links, it handles the indexing appropriately to avoid errors. The function also logs the outcome, providing a trace of the constructed URL or noting when no URL could be generated. This is particularly useful for debugging and ensuring the correct visualization links are generated and accessible.
        
        Parameters:
        deeplinks (list): A list of deep link URLs to secondary references.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        str: A concatenated URL for the Neo4J dashboard that incorporates up to three secondary reference deep links. If no deep links are provided, an empty string is returned.
        
        Notes:
        The function is currently set to handle exactly three links due to dashboard limitations. This behavior is noted as a potential area for future adjustments.
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
        '''
        Executes a query against a language model chain, returning the response and token count.

        This function interfaces with a chain of language models to perform a question-answering (QA) task. It sends the provided query along with reference data to the model, captures both the textual response and the count of tokens used in the model's reply. The token count helps in monitoring and managing usage relative to any constraints or limits. Detailed logging is performed using an optional message ID for tracking and debugging purposes.
        
        Parameters:
        query (str): The query string to be processed by the QA model.
        ref_data (str): Additional reference data that might be required by the model for generating the answer.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the model's response (str) and the token count (int) used in generating that response.
        '''
        response, tok_count = self.make_prediction(
                    self.chat_llm_chain_qa, query, "qa", msg_id, ref_data)
        return response, tok_count

    def retrieve_situational_info(self, msg_id: str = ''):
        '''
        Retrieves and returns the current date and time as a formatted string, indicating the exact moment a question was asked.

        This function constructs a formatted message providing situational information based on the current date and time. It logs this information for monitoring and debugging purposes using an optional message identifier. The function is useful for adding context to logs, particularly in scenarios where the timing of operations is crucial.
        
        Parameters:
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        str: A formatted string that contains the day of the week, date, and exact time, prefixed with a descriptive label about the situational context.
        
        Example:
        "[Situational Info] The time of asking this question is Monday, 01/01/2023 12:00:00"
        '''
        now = datetime.now()
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_of_week = now.weekday() 
        situ_info = f"\n [Situational Info] The time of asking this question is {days[day_of_week]}, {now.strftime('%d/%m/%Y %H:%M:%S')} \n"
        self.logger.info(f"MsgID={msg_id}. SITUATIONAL INFO: {situ_info}")
        return situ_info

    def query_sefaria_linker(self, text_title="", text_body="", with_text=1, debug=0, max_segments=0, msg_id: str = ''):
        '''
        Executes a query to the Sefaria Linker API by posting textual data and returns the JSON response.
        
        This function forms and sends a POST request to the Sefaria Linker API with specified text titles and bodies. It handles various request parameters and errors systematically, providing detailed logs for debugging. The function is equipped to manage HTTP errors and general exceptions, ensuring robust error handling and logging.
        
        Parameters:
        text_title (str, optional): The title of the text for which references are being sought. Defaults to an empty string.
        text_body (str, optional): The body of the text for which references are being sought. Defaults to an empty string.
        with_text (int, optional): A parameter specifying whether the response should include text; 1 for including text. Defaults to 1.
        debug (int, optional): A debug flag to provide detailed debug information in the response. Defaults to 0.
        max_segments (int, optional): The maximum number of text segments to return. Defaults to 0.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        dict or str: The JSON response from the API if successful, otherwise an error message string.
        
        Raises:
        HTTPError: If an HTTP error occurs during the API request, an HTTPError exception is raised and logged.
        '''
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
        '''
        Retrieves documents from the Sefaria Linker API based on a given query and screen_res query, applying filters to distinguish between primary and secondary sources.

        This function performs API calls to retrieve data relevant to enriched queries and processes that data based on specified filtering criteria. It is designed to support dynamic filtering of documents into primary or secondary categories, allowing for customized handling of search results.
        
        Parameters:
        screen_res (str): The screen_res query, which is used as part of the query to the Sefaria Linker.
        enriched_query (str): The enriched query string used to retrieve documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        filter_mode (str): Mode for filtering search results; valid options are 'primary' or 'secondary'. Defaults to 'primary'.
        
        Returns:
        list: A list of dictionaries, each representing a document that matches the search criteria. Each dictionary includes document details and an adjusted page rank.
        
        Raises:
        ValueError: If the provided filter_mode is not recognized, an error is raised indicating an invalid filter mode.
        '''
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

    def merge_linker_refs(self, retrieved_docs: list, p_sorted_src_rel_dict: dict, p_src_data_dict: dict, p_src_ref_dict: dict, msg_id: str = ''):
        """
        Merges new linker reference results into existing sorted source relevance dictionaries,
        data dictionaries, and reference dictionaries.

        This function is responsible for ensuring that the newly retrieved reference data from the
        Sefaria Linker is correctly integrated with existing reference data, maintaining consistency and
        accuracy across updates. Atomicity of updates is checked to ensure all updates occur simultaneously
        if all conditions are met.

        Parameters:
            page_content_total (list): Contains dictionaries with page content and metadata including URL and text.
            p_sorted_src_rel_dict (dict): Pre-existing sorted source relevance dictionary to be updated.
            p_src_data_dict (dict): Pre-existing source data dictionary that maps URLs to text data.
            p_src_ref_dict (dict): Pre-existing source reference dictionary that maps URLs to reference details.

        Returns:
            tuple: Updated dictionaries (p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict) after merging new references.
            If merging is not succesful, original value is returned.
        """

        #iterating each document in reverse order
        for document in reversed(retrieved_docs):

            # Extract necessary data to be written and to be checked
            #Extracting the URL
            new_url = 'https://www.sefaria.org/' + document['url'] if document['url'] else None
            #Extracting the page_rank score for sorting
            pr_score = float(document['page_rank']) if document['page_rank'] else None
            #Extracting the Category
            new_category = document['primaryCategory'] if document['primaryCategory'] else None
            #Extracting the Reference Part
            new_reference_part = document['url'] if document['url'] else None
            new_ref = f"Reference: {new_reference_part}. Version Title: -, Document Category: {new_category}, URL: {new_url}"
            #Extracting the english text
            new_text = ' '.join(document['en']) if document['en'] else None
            
            # Update sorted source relevance dictionary if necessary fields are satisfied
            if new_reference_part and pr_score and new_category and new_text:
                # Commit changes if fields are satisfied
                if new_url not in p_sorted_src_rel_dict:
                    p_sorted_src_rel_dict = {new_url: pr_score, **p_sorted_src_rel_dict}
                else:
                    p_sorted_src_rel_dict[new_url] = pr_score
                
                if new_url not in p_src_ref_dict:
                    p_src_ref_dict = {new_url: new_ref, **p_src_ref_dict}
                else:
                    if new_ref not in p_src_ref_dict[new_url]:
                        p_src_ref_dict[new_url] += " | " + new_ref

                if new_url not in p_src_data_dict:
                    p_src_data_dict = {new_url: new_text, **p_src_data_dict}
                else:
                    if new_text not in p_src_data_dict[new_url]:
                        p_src_data_dict[new_url] += "..." + new_text
                
                self.logger.info(f"MsgID={msg_id}. [LINKER UPDATE SUCCESSFUL] Necessary fields are satisfied for this reference: \n----new_reference_part: {new_reference_part} \n----pr_score: {pr_score} \n----new_category: {new_category} \n----new_text: {new_text} \n ")
            else:
                self.logger.info(f"MsgID={msg_id}. [LINKER UPDATE FAILED] Necessary fields are empty for this reference: \n----new_reference_part: {new_reference_part} \n----pr_score: {pr_score} \n----new_category: {new_category} \n----new_text: {new_text} \n ")
                
        #sorting it by page rank score
        p_sorted_src_rel_dict = dict(sorted(p_sorted_src_rel_dict.items(), key=lambda item: item[1], reverse=True))
        self.logger.info(f"MsgID={msg_id}. [FINAL LINKER REFERENCE MERGE OUTPUT] \n----p_sorted_src_rel_dict: {p_sorted_src_rel_dict} \n----p_src_data_dict: {p_src_data_dict} \n----p_src_ref_dict: {p_src_ref_dict}")

        return p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict

# Load basic libraries
import operator
from datetime import datetime, timedelta
from time import sleep
import os
import json
import yaml

import numpy as np
import pandas as pd
from langchain.utils.math import cosine_similarity
from langchain_core.documents import Document
# Import custom langchain modules for NLP operations and vector search
from langchain_community.vectorstores import Neo4jVector
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks import get_openai_callback
import requests

import neo4j

from VirtualHavruta.util import convert_node_to_doc, convert_vector_db_record_to_doc, \
    min_max_scaling


# Main Virtual Havruta functionalities
class VirtualHavruta:
    def __init__(self, prompts_file: str, config_file: str, logger):
        '''
        Initializes the instance with data from provided YAML files, including prompts, configurations, and reference information.
        
        This constructor method reads data from two YAML files: one containing prompts and the other containing configuration details.
        It loads the prompts and configurations into corresponding attributes.
        Additionally, it sets up the Neo4j vector index for semantic search and retrieves database configurations such as URL, username, and password.
        It initializes a logger and a pagerank lookup table based on configuration.
        Furthermore, it retrieves reference-related configurations, including filters and citation counts, and initializes prompt templates and language model instances.
        
        Parameters:
            prompts_file (str): The path to the YAML file containing prompts.
            config_file (str): The path to the YAML file containing configuration details.
            logger: The logger instance for logging information and errors.
        
        Attributes:
            prompts (dict): A dictionary containing prompts loaded from the prompts YAML file.
            config (dict): A dictionary containing configuration details loaded from the config YAML file.
            neo4j_vector (Neo4jVector): An instance of Neo4jVector for semantic search using Neo4j.
            top_k (int): The top k results to retrieve from the Neo4j database.
            neo4j_deeplink (str): The URL for the Neo4j dashboard deep link.
            logger: The logger instance used for logging information and errors.
            pr_table (DataFrame): A pandas DataFrame containing pagerank lookup table data.
            primary_source_filter (list): A list of primary source filters for reference data.
            num_primary_citations (int): The number of primary citations to retrieve.
            num_secondary_citations (int): The number of secondary citations to retrieve.
            linker_primary_source_filter (list): A list of primary source filters specific to linker references.
        
        Methods:
            initialize_prompt_templates(): Initializes prompt templates based on configuration data.
            initialize_llm_instances(): Initializes language model instances based on configuration data.
        '''
        with open(prompts_file, 'r') as f:
            self.prompts = yaml.safe_load(f)
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize Neo4j vector index and retrieve DB configs
        self.model_api = self.config['openai_model_api']
        config_emb_db = self.config['database']['embed']
        self.neo4j_vector = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(model=self.model_api['embedding_model']),
            index_name="index",
            url=config_emb_db['url'],
            username=config_emb_db['username'],
            password=config_emb_db['password'],
        )
        self.top_k = config_emb_db['top_k']
        self.neo4j_deeplink = self.config['database']['kg']['neo4j_deeplink']

        # Initiate logger
        self.logger = logger

        # Retrieve reference configs
        refs = self.config['references']
        linker_references = self.config['linker_references']
        self.primary_source_filter = refs['primary_source_filter']
        self.num_primary_citations = refs['num_primary_citations']
        self.num_secondary_citations = refs['num_secondary_citations']
        self.num_primary_citations_linker = linker_references['num_primary_citations']
        self.num_secondary_citations_linker = linker_references['num_secondary_citations']
        self.linker_primary_source_filter = linker_references['primary_source_filter']
        
        # Initialize prompt templates and LLM instances
        self.initialize_prompt_templates()
        self.initialize_llm_instances()

    def initialize_prompt_templates(self):
        '''
        Initializes prompt templates for various chat interactions and updates class attributes accordingly.

        This function initializes prompt templates for different chat interaction categories such as anti-attack, adaptor, editor, optimization, and classification.
        It iterates over a list of categories, creates a prompt template for each category based on the 'system' template, and updates the class attributes with the generated templates.
        Additionally, it creates a separate prompt template for the QA (question-answering) category, including reference data.

        '''
        no_ref_categories = ['anti_attack', 'adaptor', 'editor', 'optimization']
        ref_categories = ['classification', 'qa', 'selector']
        no_ref_prompts = {'prompt_'+cat: self.create_prompt_template('system', cat) for cat in no_ref_categories}
        ref_prompts = {'prompt_'+cat: self.create_prompt_template('system', cat, True) for cat in ref_categories}
        self.__dict__.update(no_ref_prompts)
        self.__dict__.update(ref_prompts)

    def create_prompt_template(self, category: str, template: str, ref_mode: bool = False) -> ChatPromptTemplate:
        '''
        Creates a prompt template for chat interactions based on a given category and template, optionally incorporating reference data.
        
        This function generates a prompt template suitable for chat interactions by combining system messages with a human message template.
        It constructs the human message template dynamically based on whether reference data is required, incorporating it if the `ref_mode` parameter is set to True.
        The resulting prompt template is encapsulated in a `ChatPromptTemplate` object, which includes both system and human message components.
        
        Parameters:
            category (str): The category of the prompt template, specifying the type of interaction or task.
            template (str): The specific template within the category to be used for constructing the prompt.
            ref_mode (bool, optional): A flag indicating whether reference data should be included in the prompt; defaults to False.
        
        Returns:
            ChatPromptTemplate: A `ChatPromptTemplate` object containing the system message and human message components necessary for chat interactions.
        
        Example:
            create_prompt_template("qa", "default", ref_mode=True) returns a `ChatPromptTemplate` object with a system message from the 'qa' category and a human message template that includes reference data.
        '''
        system_message = SystemMessage(content=self.prompts[category][template])
        human_template = f"Question: {{human_input}}{' Reference Data: {ref_data}' if ref_mode else ''}."
        return ChatPromptTemplate.from_messages([
            system_message,
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def initialize_llm_instances(self):
        '''
        Initializes multiple language model instances on a class instance based on configuration parameters.
        
        This method initializes multiple language model instances on a class instance by loading required configuration parameters from configuration files.
        It retrieves information about the OpenAI model API and language model chain setups.
        For each model setup specified in the 'llm_chain_setups' section of the configuration, it creates a corresponding language model instance.
        If a model name ends with '_json', it includes additional keyword arguments to specify JSON response format.
        
        Parameters: None
        
        Returns: None
        '''
        chain_setups = self.config['llm_chain_setups']
        
        # Adding a condition to include json kwargs for models ending with '_json'
        for model_name, suffixes in chain_setups.items():
            model_kwargs = {"response_format": {"type": "json_object"}} if model_name.endswith('_json') else {}
            model_key = model_name.replace('_json', '')  # Removes the '_json' suffix for lookup in model_api
            setattr(self, model_name, ChatOpenAI(
                temperature=self.model_api.get(f"{model_key}_temperature", None),
                model=self.model_api.get(model_key, None),
                model_kwargs=model_kwargs
            ))
            self.initialize_llm_chains(getattr(self, model_name), suffixes)

    def initialize_llm_chains(self, model, suffixes):
        '''
        Initializes multiple language model chains on a class instance, each configured with a specific prompt template and suffix.
        
        This function dynamically creates and assigns language model chain objects to attributes of a class instance.
        It uses a base model and a list of suffixes to generate attribute names and corresponding prompt templates.
        Each chain is initialized with the same model but different prompt templates, which are assumed to be predefined as attributes on the class instance.
        This approach facilitates the management and use of multiple specialized tasks, such as QA, optimization, and adaptation, each requiring different prompt configurations.
        
        Parameters:
        model (LanguageModel): The language model to be used for all chains.
        suffixes (list of str): A list of suffix identifiers that correspond to different tasks or configurations. These suffixes are used to form both the attribute names for the chains and to retrieve corresponding prompt templates from the class instance.

        Example:
            initialize_llm_chains(getattr(self, model_name), suffixes)
        '''
        for suffix in suffixes:
            setattr(self, f"chat_llm_chain_{suffix}",
                    self.create_llm_chain(model, getattr(self, f"prompt_{suffix}")))

    def create_llm_chain(self, llm, prompt_template):
        '''
        Creates and returns an instance of a language model chain configured with a specified language model and prompt template.
        
        This function initializes a language model chain using the provided language model and prompt template.
        It sets the verbosity level to 'False' by default, which minimizes logging or debug output from the chain itself.
        The resulting object is designed to facilitate customized interactions with the language model based on the specified prompt structure, enhancing the flexibility and applicability of the model for various tasks.
        
        Parameters:
        llm (LanguageModel): The language model to be used in the chain.
        prompt_template (str): The template string that defines the structure and content of prompts to be sent to the language model.
        
        Returns:
        LLMChain: An instance of a language model chain configured with the given language model and prompt template.

        Example:
        create_llm_chain(model, getattr(self, f"prompt_{suffix}")))
        '''
        return LLMChain(llm=llm, prompt=prompt_template, verbose=False)

    def make_prediction(self, chain, query: str, action: str, msg_id: str = '', ref_data: str = ''):
        '''
        Executes a prediction using a specified language model chain, providing logging and token tracking.

        This function interfaces with a language model chain to perform a specific action (e.g., QA, optimization, editing) based on the provided query and optional reference data.
        It measures the number of tokens used in the process using a callback mechanism and logs both successful results and errors.
        The function is designed to handle both scenarios where reference data is and is not provided, optimizing its request to the model accordingly.
        
        Parameters:
        chain (LanguageModelChain): The specific language model chain used for prediction.
        query (str): The input query string for which the prediction is needed.
        action (str): The type of action the model is performing, used for logging.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        ref_data (str, optional): Additional reference data to be included in the prediction request; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the result of the prediction (str) and the number of tokens used (int).
        
        Raises:
        Exception: Catches and logs any exceptions that occur during the prediction process, including token expenditure.

        Example:
        make_prediction(self.chat_llm_chain_anti_attack, query, "ANTI-ATTACK", msg_id)
        '''
        with get_openai_callback() as cb:
            try: 
                res = chain.predict(human_input=query, ref_data=ref_data) if ref_data else chain.predict(human_input=query)
                self.logger.info(f"MsgID={msg_id}. [INFERENCE] Spent {cb.total_tokens} tokens for {action}. Query={query}. Reference data={ref_data}. Result={res}.")
            except Exception as e:
                self.logger.error(f"MsgID={msg_id}. [INFERENCE] Spent {cb.total_tokens} tokens for {action} but failed. Error is {e}.")
                res = ''
            return res, cb.total_tokens

    def anti_attack(self, query: str, msg_id: str = ''):
        '''
        Analyzes a query for potential attacks using a language model chain specialized in anti-attack tasks, returning the detection status, explanation, and token count.
        
        This function submits a query to an anti-attack model, which assesses the text for elements that might constitute an attack or harmful content.
        The model's response is expected to include a detection status and an explanation, separated by a special delimiter.
        If the parsing of the response fails, the function logs the error and defaults the detection to 'N' (No) with an empty explanation.
        This ensures reliable operation even in cases of unexpected model output or processing errors.
        
        Parameters:
        query (str): The query string to be analyzed for potential attacks.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the detection status (str), an explanation (str), and the token count (int) used during the analysis.
        
        Raises:
        Exception: Catches and logs any exception that occurs during response parsing, setting default values for the detection status and explanation.

        Example:
        detection, explanation, tok_count = vh.anti_attack(query, msgid)
        '''
        adv_res, tok_count = self.make_prediction(self.chat_llm_chain_anti_attack, query, "ANTI-ATTACK", msg_id)
        try:
            detection, explanation = adv_res.split('@SEP@')
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. [Anti-Attack] Error occurred during attack detection: {e}.")
            detection, explanation = 'N', ''
        return detection, explanation, tok_count

    def adaptor(self, query: str, msg_id: str = ''):
        '''
        Processes a query using a language model chain optimized for adaptation tasks, returning the adapted result along with the token count.

        This function sends a query to an adaptation-specific model, which modifies the query to fit particular contexts or requirements.
        It retrieves the adapted text and the number of tokens used in the model's response.
        The function is useful for tasks requiring contextual modifications or specific formatting.
        It also logs each transaction with an optional message identifier, aiding in monitoring and debugging processes.
        
        Parameters:
        query (str): The query string to be adapted by the model.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the adapted text (str) and the token count (int) used in generating that adapted text.

        Example:
        screen_res, tok_count = vh.adaptor(query, msgid)
        '''
        adp_res, tok_count = self.make_prediction(self.chat_llm_chain_adaptor, query, "ADAPTATION", msg_id)
        return adp_res, tok_count

    def editor(self, query: str, msg_id: str = ''):
        '''
        Performs editing on a given query using a language model chain optimized for editing tasks, returning the edited result along with the token count.

        This function sends a query to an editing-optimized model, which processes and refines the text to improve clarity, style, or correctness.
        The function captures the edited output and the count of tokens used by the model, facilitating usage tracking and evaluation.
        It also logs the operation using an optional message identifier, enhancing traceability and assisting with debugging.
        
        Parameters:
        query (str): The query string to be edited by the model.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the edited text (str) and the token count (int) used in generating that edited text.

        Example:
        screen_res, tok_count = vh.editor(query, msgid)
        '''
        edit_res, tok_count = self.make_prediction(self.chat_llm_chain_editor, query, "EDITING", msg_id)
        return edit_res, tok_count
        
    def optimizer(self, query: str, msg_id: str = ''):
        '''
        Optimizes a query using a chain of language models dedicated to prompt optimization, extracting various components from the optimization results.

        This function submits a query to an optimization model, which processes the query and returns structured optimization results.
        These results are expected to contain components such as translation, key concepts, elaboration, quotation, challenges, and potential directions.
        The function decodes the JSON response, extracts these components, and returns them along with the token count used in the operation.
        Errors during JSON processing are logged, and default values are used if an error occurs, ensuring the function remains robust across different scenarios.
        
        Parameters:
        query (str): The query string to be optimized by the model.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the translated query, extracted key concepts, elaboration details, any related quotations, challenges identified, potential directions proposed, and the total token count used during the optimization process.
        
        Raises:
        Exception: Catches and logs any exception that occurs during the JSON parsing and sets all output components to empty strings as a fallback.

        Example:
        translation, extraction, elaboration, quotation, challenge, proposal, tok_count = vh.optimizer(screen_res, msgid)
        '''
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
            self.logger.error(f"MsgID={msg_id}. [OPTIMIZATION] Error occurred during PROMPT OPTIMIZATION: {e}.")
            translation = extraction = elaboration = quotation = challenge = proposal = ''
        return translation, extraction, elaboration, quotation, challenge, proposal, tok_count

    def retrieve_docs(self, query: str, msg_id: str = '', filter_mode: str='primary'):
        '''
        Retrieves documents that match a specified query and filters them based on whether they are primary or secondary sources, using a similarity search.

        This function performs a similarity search based on the provided query and retrieves documents that either match the characteristics of primary or secondary sources as defined by a filter set.
        The results are filtered by checking each document's metadata against a predefined set of source filters.
        The function logs the process to ensure transparency and is equipped to handle errors related to invalid filter modes, raising a ValueError if necessary.
        
        Parameters:
        query (str): The query string used to search for relevant documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        filter_mode (str): The mode to filter the search results by 'primary' or 'secondary' to determine the relevance of the sources.
        
        Returns:
        retrieval_res: A list of documents that meet the criteria of the specified filter mode, either as primary or secondary sources.
        
        Raises:
        ValueError: If an invalid filter_mode is provided, an exception is raised to indicate the error.

        Example:
        primary_retrieval_result = vh.retrieve_docs(query, msgid, 'primary')
        '''
        self.logger.info(f"MsgID={msg_id}. [RETRIEVAL] Simple semantic search at work. Retrieving {filter_mode} references using this query: {query}")
        # Convert primary_source_filter to a set for efficient lookup
        retrieved_docs = self.neo4j_vector.similarity_search_with_relevance_scores(
            query, self.top_k,
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

    def retrieve_docs_metadata_filtering(self, query: str, msg_id: str = '', metadata_fiter: dict|None=None):
        '''
        Retrieves documents that match a specified query and filters them based on their metadata, using a similarity search.

        This function performs a similarity search based on the provided query and retrieves documents that match the metadata conditions as defined by a metadata_filter.
        The results are filtered by applying the metadata filters during semantic search.
        The function logs the process to ensure transparency.
        
        Parameters:
        query (str): The query string used to search for relevant documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        metadata_fiter (dict): The metadata filter dictionary used to filter the search results during semantic search.
        
        Returns:
        list: A list of documents that meet the criteria of the specified metadata filter.

        Example:
        p_retrieval_res = vh.retrieve_docs_metadata_filtering(query, msgid, metadata_filter)
        '''
        self.logger.info(f"MsgID={msg_id}. [RETRIEVAL] Metadata filtering at work. Retrieving references using this query: {query} and this metadata filter {metadata_fiter}")
        # Convert primary_source_filter to a set for efficient lookup
        retrieved_res = self.neo4j_vector.similarity_search_with_relevance_scores(
            query, self.top_k, filter=metadata_fiter
            )
        return retrieved_res
    
    def retrieve_nodes_matching_linker_results(self, linker_results: list[dict], msg_id: str = '', filter_mode: str = 'primary',
                                               url_prefix: str = "https://www.sefaria.org/") -> list[Document]:
        '''
        Retrieve nodes corresponding to linker results.

        Given linker results, find and return the corresponding nodes in the graph database.
        There is a one-to-many relationship between a linker result and graphs in the graph db.

        Parameters:
        linker_results : list
            Results from the linker API.
        msg_id : str, optional
            Identifier for Slack bot messages, by default ''.
        filter_mode : str, optional
            Mode for filtering search results; valid options are 'primary' or 'secondary'. Defaults to 'primary'.
        url_prefix : str, optional
            Adds domain if missing, by default "https://www.sefaria.org/".

        Returns:
        list
            A list of documents matching the linker results.

        Example:
        res = vh.retrieve_nodes_matching_linker_results(linker_results, msg_id, filter_mode=filter_mode)
        '''

        urls_linker_results = list({url_prefix +linker_res["url"] if not linker_res["url"].startswith("http") else linker_res["url"]
                                    for linker_res in linker_results})
        self.logger.info(f"MsgID={msg_id}. [LINKER-GRAGH RETRIEVAL] Retrieving graph nodes using linker URLs: {urls_linker_results}")
        nodes_linker: list[Document] = self.query_graph_db_by_url(urls=urls_linker_results)
        url_to_node = {}
        for node in nodes_linker:
            if (url:=node.metadata["url"])  not in url_to_node:
                url_to_node[url] = node
            else:
                url_to_node[url].metadata["source"] += " | " + node.metadata["source"]
        self.logger.info(f"MsgID={msg_id}. [LINKER-GRAGH RETRIEVAL] Graph nodes retrieved using linker URLs: {['URL='+url+' SOURCE='+node.metadata["source"] for url, node in url_to_node.items()]}")
        return list(url_to_node.values())
    
    def get_retrieval_results_knowledge_graph(self, url: str, direction: str, order: int, score_central_node: float, filter_mode_nodes: str|None = None, msg_id: str = '') -> list[tuple[Document, float]]:
        '''
        Given a URL, query the graph database for the neighbors of the node with that URL.

        Scores the neighbors based on their distance to the central node.

        Parameters:
        url : str
            The URL of the central node.
        direction : str
            The direction of the edges between nodes, one of 'incoming', 'outgoing', 'both_ways'. In the Sefaria KG, edges point from newer to older references.
            'incoming' searches for newer references, 'outgoing' for older references, and 'both_ways' for both.
        order : int
            Order of neighbors (number of hops) to include, between 1 and n.
        score_central_node : float, optional
            Score of the central node, by default 6.0.
        filter_mode_nodes : str, optional
            Mode for filtering search results, if provided; valid options are 'primary' or 'secondary'. Defaults to None for no filter.

        Returns:
        list
            A list of tuples, each containing a document and its score.

        Example:
        res = vh.get_retrieval_results_knowledge_graph(
            url=top_node.metadata["url"],
            direction=self.config["database"]["kg"]["direction"],
            order=self.config["database"]["kg"]["order"],
            filter_mode_nodes=filter_mode_nodes,
            score_central_node=6.0,
            msg_id=msg_id
        )
        '''
        self.logger.info(f"MsgID={msg_id}. [GRAGH NEIGHBOR RETRIEVAL] Starting get_retrieval_results_knowledge_graph.")
        nodes_distances = self.get_graph_neighbors_by_url(url, direction, order, filter_mode_nodes=filter_mode_nodes, msg_id=msg_id)
        nodes = [node for node, _ in nodes_distances]
        docs =  [convert_node_to_doc(node) for node in nodes]
        distances = [distance for _, distance in nodes_distances]
        scores = [self.score_document_by_graph_distance(distance, start_score=score_central_node, score_decrease_per_hop=0.1) for distance in distances]
        return list(zip(docs, scores, strict=True))

    def score_document_by_graph_distance(self, n_hops: int, start_score: float, score_decrease_per_hop: float) -> float:
        '''
        Score a document by the number of hops from the central node.

        Parameters:
        n_hops : int
            Number of hops from the central node.
        start_score : float
            Score of the central node.
        score_decrease_per_hop : float
            Decrease of score per hop.

        Returns:
        float
            The calculated score.

        Example:
        scores = [vh.score_document_by_graph_distance(
            distance, 
            start_score=score_central_node, 
            score_decrease_per_hop=0.1
        ) for distance in distances]
        '''
        return max(start_score - n_hops * score_decrease_per_hop, 0.0)

    def get_graph_neighbors_by_url(self, url: str, relationship: str, depth: int, filter_mode_nodes: str|None = None, msg_id: str = '') -> list[tuple["Node", int]]:
        '''
        Given a URL, query the graph database for the neighbors of the node with that URL.

        Parameters:
        url : str
            The URL of the central node.
        relationship : str
            The direction of the edges between nodes, one of 'incoming', 'outgoing', 'both_ways'. In the Sefaria KG, edges point from newer to older references.
            'incoming' searches for newer references, 'outgoing' for older references, and 'both_ways' for both.
        depth : int
            Degree of neighbors to include, between 1 and n.

        Returns:
        list
            A list of (node, distance) tuples, where distance is the number of hops from the central node.

        Example:
        res = vh.get_graph_neighbors_by_url(
            url, 
            direction, 
            order, 
            filter_mode_nodes=filter_mode_nodes, 
            msg_id=msg_id
        )
        '''
        self.logger.info(f"MsgID={msg_id}. [GRAGH NEIGHBOR RETRIEVAL] Retrieving graph neighbors for url: {url}.")
        assert relationship in ["incoming", "outgoing", "both_ways"]
        start_node_operator: str = "<-" if relationship == "incoming" else "-"
        related_node_operator: str = "->" if relationship == "outgoing" else "-"
        nodes = []
        primary_doc_categories = [category.replace("Document Category: ", "") for category in self.primary_source_filter]
        query_params: dict = {"url": url, "primaryDocCategories": primary_doc_categories}
        for i in range(1, depth + 1):
            source_filter = f'AND {"NOT" if filter_mode_nodes == "secondary" else ""} neighbor.primaryDocCategory IN $primaryDocCategories' if filter_mode_nodes else ''
            query = f"""
            MATCH (start:Records {{url: $url}})
            WITH start
            MATCH (start){start_node_operator}[:FROM_TO*{i}]{related_node_operator}(neighbor)
            WHERE neighbor <> start
            {source_filter}
            RETURN DISTINCT neighbor, {i} AS depth
            """
            with neo4j.GraphDatabase.driver(self.config["database"]["kg"]["url"], auth=(self.config["database"]["kg"]["username"], self.config["database"]["kg"]["password"])) as driver:
                neighbor_nodes, _, _ = driver.execute_query(
                query,
                parameters_=query_params,
                database_=self.config["database"]["kg"]["name"],)
            nodes.extend(neighbor_nodes)
        self.logger.info(f"MsgID={msg_id}. [GRAGH NEIGHBOR RETRIEVAL] Retrieved {len(nodes)} graph neighbors.")
        return nodes

    def query_graph_db_by_url(self, urls: list[str]) -> list[Document]:
        '''
        Given a list of URLs, query the graph database for the nodes with those URLs.

        Note that there is a one-to-many relationship between URLs and documents in the vector database,
        due to different sources for the same URL.

        Returns the nodes in a document-compatible type.

        Parameters:
        urls : list
            A list of URLs of the documents.

        Returns:
        list
            A list of documents.

        Example:
        nodes_linker: list[Document] = vh.query_graph_db_by_url(urls=urls_linker_results)
        '''
        query_parameters = {"urls": urls}
        query_string="""
        MATCH (n:Records)
        WHERE any(substring IN $urls WHERE n.url = substring)
        RETURN n
        """
        with neo4j.GraphDatabase.driver(self.config["database"]["kg"]["url"], auth=(self.config["database"]["kg"]["username"], self.config["database"]["kg"]["password"])) as driver:
            nodes, _, _ = driver.execute_query(
            query_string,
            parameters_=query_parameters,
            database_=self.config["database"]["kg"]["name"],)
        return [convert_node_to_doc(node) for node in nodes]

    def select_reference(self, query: str, retrieval_res, msg_id: str = ''):
        '''
        Based on the provided query and retrieval_res, select useful references using a chained language model, returning the selected retrieval_res and token count.

        This function selects retrieval results based on a language model specifically tuned for selection tasks.
        It captures the selected retrieval results, which are expected to be a list of documents, and the count of tokens used by the model. 
        If the function's output cannot be converted to a list of documents due to an error, the function logs the error and defaults the selected results to [].
        This ensures robust error handling and maintains the integrity of the selection process under all conditions.
        
        Parameters:
        query (str): The query string to be referred to by the model.
        retrieval_res (list): A list of retrieved documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the selected retrieval results (list of documents) and the token count (int) used in generating that result.
        
        Raises:
        Exception: Catches and logs any exception that occurs during the selection process, defaulting the result to [] and 0.

        Example:
        seed_chunks, token_count = vh.select_reference(enriched_query, seed_chunks, msg_id=msg_id)
        '''
        
        try:
            # Construct reference data string        
            conc_ref_data = ''
            for n, res in enumerate(retrieval_res):
                if isinstance(res, tuple):
                    d, _ = res
                else:
                    d = res
                # Concatenate reference data and its source
                numbered_ref_data = f'#{n}# {d.page_content}... --Origin of this {d.metadata["source"]} '
                conc_ref_data += numbered_ref_data
            selected_idx, tok_count = self.selector(query, conc_ref_data, msg_id)
            selected_retrieval_res = [retrieval_res[i] for i in selected_idx]
        except Exception as e:
            self.logger.error(
                f"MsgID={msg_id}. Reference selection result was set to []. Error message is {e}."
            )
            selected_retrieval_res = []
            tok_count = 0
        
        return selected_retrieval_res, tok_count

    def sort_reference(self, scripture_query: str, enriched_query: str, retrieval_res, filter_mode: str|None = 'primary', msg_id: str = ''):
        '''
        Sorts and processes retrieval results for references based on their relevance to a given query, considering both primary and secondary filtering modes.
        
        This function processes a set of retrieval results, classifying each result for relevance and calculating a composite relevance score based on classification results, similarity scores, and, for primary references, PageRank scores.
        It also consolidates results with the same URL to avoid duplication, ensuring that the most relevant and comprehensive content is retained.
        The function logs each step for transparency and debugging purposes and returns dictionaries containing sorted relevance data, source data, and reference details, along with the total count of tokens used in processing.
        
        Parameters:
        scripture_query (str): The query string against which references are being sorted and classified.
        enriched_query (str): The enriched query string used to retrieve documents.
        retrieval_res (iterable): An iterable of tuples containing reference data objects and similarity scores.
        filter_mode: set if all retrieval results are from either primary or secondary sources, set to None if both are present. Defaults to 'primary'.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing sorted source relevance dictionary, source data dictionary, source reference dictionary, and the total token count used during the process.
        
        Notes:
        The function is robust to variations in data and manages complex scenarios where multiple references may have the same URL but different content or sources. It effectively manages and logs all operations to ensure data integrity and traceability.

        Example:
                p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, tok_count = vh.sort_reference(
                scripture_query=scripture_query,
                enriched_query=enriched_query,
                retrieval_res=sel_p_retrieval_res,
                msg_id=msgid,
                filter_mode=primary_filter_mode
            )
        '''
        total_tokens = 0

        documents = [d for d, _ in retrieval_res]
        semantic_similarity_scores = [sim_score for _, sim_score in retrieval_res]
        sorted_docs, sorted_ranking_scores, token_count = self.rank_documents(documents,
                                                                              enriched_query=enriched_query,
                                                                              scripture_query=scripture_query,
                                                                              semantic_similarity_scores=semantic_similarity_scores,
                                                                              filter_mode=filter_mode,
                                                                              msg_id=msg_id)
        total_tokens += token_count

        retrieval_res_ranked = list(zip(sorted_docs, sorted_ranking_scores))
        sorted_src_rel_dict, src_data_dict, src_ref_dict = self.merge_references_by_url(retrieval_res_ranked, msg_id=msg_id)
        return sorted_src_rel_dict, src_data_dict, src_ref_dict, total_tokens

    def merge_references_by_url(self, retrieval_res: list[tuple[Document, float]], msg_id: str = '') -> tuple[dict, dict, dict]:
        '''
        Merge chunks with the same URL.

        This can occur for two reasons:
        1. Different graph nodes with the same URL.
        2. The same graph node split into multiple chunks.

        Parameters:
        retrieval_res : list
            A list of (document, ranking_score) tuples.
        msg_id : str, optional
            Slack message ID, by default "".

        Returns:
        tuple
            A tuple containing sorted source relevance dictionary, source data dictionary, and source reference dictionary.

        Example:
        sorted_src_rel_dict, src_data_dict, src_ref_dict = vh.merge_references_by_url(retrieval_res_ranked, msg_id=msg_id)
        '''
        src_data_dict = {}
        src_ref_dict = {}
        src_rel_dict = {}
        # Iterate over each item in the retrieval results
        for (d, rel_score) in retrieval_res:
            # If the URL is not already in src_data_dict, add all reference information
            if d.metadata["url"] not in src_data_dict:
                src_data_dict[d.metadata["url"]] = d.page_content
                src_ref_dict[d.metadata["url"]] = d.metadata["source"]
                src_rel_dict[d.metadata["url"]] = rel_score
            else:
                # If the URL is already present, handle different versions or sources with the same URL
                existing_content = src_data_dict[d.metadata["url"]]
                # Concatenate page content for the same URL
                src_data_dict[d.metadata["url"]] = "...".join([existing_content, d.page_content])

                # Avoid duplicate source listings by separating with a pipe "|"
                existing_ref = src_ref_dict[d.metadata["url"]]
                existing_ref_list = existing_ref.split(" | ")
                if d.metadata["source"] not in existing_ref_list:
                    src_ref_dict[d.metadata["url"]] = " | ".join([existing_ref, d.metadata["source"]])

                # Update the relevance score with the maximum score between existing and new
                existing_rel_score = src_rel_dict[d.metadata["url"]]
                src_rel_dict[d.metadata["url"]] = max(existing_rel_score, rel_score)

        # Sort the source relevance dictionary based on scores in descending order
        sorted_src_rel_dict = dict(
            sorted(src_rel_dict.items(), key=operator.itemgetter(1), reverse=True)
        )
        self.logger.info(f"MsgID={msg_id}. [MERGE REFERENCE] sorted_src_rel_dict={sorted_src_rel_dict}, src_data_dict={src_data_dict}, src_ref_dict={src_ref_dict}.")
        # Return the sorted source relevance dictionary, source data dictionary, source reference dictionary, and token count
        return sorted_src_rel_dict, src_data_dict, src_ref_dict

    def selector(self, query: str, ref_data: str, msg_id: str = ""):
        '''
        Based on the provided query and numbered reference data, select useful references using a chained language model, returning the selected indices and token count.

        This function sends a query and reference data to a language model specifically tuned for selection tasks.
        It captures the selection result, which is expected to be a list of numerical values, and the count of tokens used by the model. 
        If the model's output cannot be converted to a list of integers due to an error, the function logs the error and defaults the selection to [].
        This ensures robust error handling and maintains the integrity of the selection process under all conditions.
        
        Parameters:
        query (str): The query string to be referred to by the model.
        ref_data (str): Reference data related to the query that may be used to answer the query.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the selected indices (list of int) and the token count (int) used in generating that result.
        
        Raises:
        Exception: Catches and logs any exception that occurs during the selection process, defaulting the result to [].

        Example:
        selected_idx, tok_count = vh.selector(query, conc_ref_data, msg_id)
        '''
        
        response, tok_count = self.make_prediction(
            self.chat_llm_chain_selector, query, "SELECTOR", msg_id, ref_data
        )
        try:
            if response.strip() == ',':
                selected_idx = []
            else:
                selected_idx = [int(x) for x in response.split(',') if x]
        except Exception as e:
            self.logger.error(
                f"MsgID={msg_id}. LLM SELECTOR result was set to []. Error message is {e}."
            )
            selected_idx = []
        
        return selected_idx, tok_count

    def classification(self, query: str, ref_data: str, msg_id: str = ''):
        '''
        Classifies the provided query and reference data using a chained language model, returning the classification result and token count.

        This function sends a query and reference data to a language model specifically tuned for classification tasks.
        It captures the classification result, which is expected to be a numerical value, and the count of tokens used by the model. 
        If the model's output cannot be converted to an integer due to an error, the function logs the error and defaults the classification to 0.
        This ensures robust error handling and maintains the integrity of the classification process under all conditions.
        
        Parameters:
        query (str): The query string to be classified by the model.
        ref_data (str): Reference data related to the query that may influence the classification.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the classification result (int) and the token count (int) used in generating that result.
        
        Raises:
        Exception: Catches and logs any exception that occurs during the classification conversion process, defaulting the result to 0.

        Example:
        ref_class, token_count = vh.classification(query=query, ref_data=ref_data, msg_id=msg_id)
        '''
        # Classifiy the data with LLM
        ref_class, tok_count = self.make_prediction(
                    self.chat_llm_chain_classification, query, "CLASSIFICATION", msg_id, ref_data)
        try:
            ref_class = int(ref_class)
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. [CLASSIFICATION] Result was set to 0. Error message is {e}.")
            ref_class = 0
        return ref_class, tok_count

    def generate_ref_str(self, sorted_src_rel_dict, src_data_dict, src_ref_dict, msg_id: str = '', ref_mode: str = 'primary', n_citation_base: int = 0, is_linker_search: bool = False) -> str:
        '''
        Constructs formatted reference strings and citation lists based on the source relevance and data dictionaries, with specific handling for primary and secondary references.
        
        This function dynamically generates reference content and citation indices from sorted relevance data, differentiated by primary and secondary modes.
        It assembles detailed reference strings, citation lists, and deep links for secondary references if applicable.
        The function allows for continuation of citation numbering from a specified base, which is useful in documents where references span multiple sections or components.
        
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

        Example:
        example = vh.generate_ref_str(p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, msgid, 'primary')
        '''
        # Determine the starting citation number and how many citations to include
        n_citation_base = 0 if ref_mode == 'primary' else n_citation_base
        if is_linker_search:
            num_citations = self.num_primary_citations_linker if ref_mode == 'primary' else self.num_secondary_citations_linker
        else:
            num_citations = self.num_primary_citations if ref_mode == 'primary' else self.num_secondary_citations

        # Lists to hold parts of the final strings
        ref_data_parts = []
        citation_parts = []
        deeplinks = []
        n_citation = 0

        # Process only the needed citations
        for n, (k, rel_score) in enumerate(sorted_src_rel_dict.items()):
            if num_citations > 0 and n >= num_citations:
                break

            n_citation = n_citation_base + n + 1
            ref_data_parts.append(f"\n #Reference {n_citation}# {src_data_dict[k]}... --Origin of this {src_ref_dict[k]} \n")
            citation_parts.append(f"\n{n_citation}. {src_ref_dict[k]}")

            # Additional actions for secondary references
            if ref_mode == 'secondary':
                deeplinks.append(k)

            self.logger.info(
                    f"MsgID={msg_id}. [GENERATE REFERENCE STRING] Included this reference for {ref_mode} references: {k}. Relevance score = {sorted_src_rel_dict[k]}."
                )

        # Join the parts into final strings
        conc_ref_data = ''.join(ref_data_parts)
        citations = ''.join(citation_parts)

        return conc_ref_data, citations, deeplinks, n_citation

    def generate_kg_deeplink(self, deeplinks, msg_id: str = ''):
        '''
        Generates a Knowledge Graph (KG) deep link URL by concatenating up to the first three secondary reference URLs provided.
        
        This function constructs a URL for the Neo4J dashboard by using up to three deep links from the provided list.
        If there are fewer than three deep links, it handles the indexing appropriately to avoid errors.
        The function also logs the outcome, providing a trace of the constructed URL or noting when no URL could be generated.
        This is particularly useful for debugging and ensuring the correct visualization links are generated and accessible.
        
        Parameters:
        deeplinks (list): A list of deep link URLs to secondary references.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        str: A concatenated URL for the Neo4J dashboard that incorporates up to three secondary reference deep links. If no deep links are provided, an empty string is returned.
        
        Notes:
        The function is currently set to handle exactly three links due to dashboard limitations. This behavior is noted as a potential area for future adjustments.

        Example:
        neo4j_deeplink = vh.generate_kg_deeplink(deeplinks, msgid)
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
            self.logger.info(f"MsgID={msg_id}. [KG DEEP LINK] Created KG deep link for secondary references: {neo4j_deeplink}.")
        else:
            neo4j_deeplink = ""
            self.logger.info(f"MsgID={msg_id}. [KG DEEP LINK] Empty KG deep link for secondary references.")
        return neo4j_deeplink

    def qa(self, query: str, ref_data: str, msg_id: str = ''):
        '''
        Executes a query against a language model chain, returning the response and token count.

        This function interfaces with a chain of language models to perform a question-answering (QA) task. 
        It sends the provided query along with reference data to the model, captures both the textual response and the count of tokens used in the model's reply. 
        The token count helps in monitoring and managing usage relative to any constraints or limits.
        Detailed logging is performed using an optional message ID for tracking and debugging purposes.
        
        Parameters:
        query (str): The query string to be processed by the QA model.
        ref_data (str): Additional reference data that might be required by the model for generating the answer.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        
        Returns:
        tuple: A tuple containing the model's response (str) and the token count (int) used in generating that response.

        Example:
        response, tok_count = vh.qa(query, ref_data, msgid)
        '''
        response, tok_count = self.make_prediction(
                    self.chat_llm_chain_qa, query, "qa", msg_id, ref_data)
        return response, tok_count

    def retrieve_situational_info(self, msg_id: str = ''):
        '''
        Retrieves and returns the current date and time as a formatted string, indicating the exact moment a question was asked.

        This function constructs a formatted message providing situational information based on the current date and time.
        It logs this information for monitoring and debugging purposes using an optional message identifier.
        The function is useful for adding context to logs, particularly in scenarios where the timing of operations is crucial.
        
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
        situ_info = f"[Situational Info] The time of asking this question is {days[day_of_week]}, {now.strftime('%d/%m/%Y %H:%M:%S')}."
        self.logger.info(f"MsgID={msg_id}. [SITUATIONAL INFO] Retrieved current situation: {situ_info}")
        return situ_info

    def query_sefaria_linker(self, text_title="", text_body="", with_text=1, debug=0, max_segments=0, msg_id: str = ''):
        '''
        Executes a query to the Sefaria Linker API by posting textual data and returns the JSON response.
        
        This function forms and sends a POST request to the Sefaria Linker API with specified text titles and bodies.
        It handles various request parameters and errors systematically, providing detailed logs for debugging.
        The function is equipped to manage HTTP errors and general exceptions, ensuring robust error handling and logging.
        
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

        Example:
        result = vh.query_sefaria_linker(text_title=screen_res, text_body=enriched_query, msg_id=msg_id)
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
            self.logger.info(f"MsgID={msg_id}. [LINKER RETRIEVAL] Retrieving linker references using this json: {data}")
            response = requests.post(api_url, headers=headers, params=params, json=data)
            self.logger.info(f"MsgID={msg_id}. [LINKER RETRIEVAL] Sefaria linker query response: {response}. {response.json()}.")
            response.raise_for_status()  # Handles HTTP errors by raising an HTTPError exception for bad requests
            # response.json() will return the JSON response for a successful request
            return response.json()
        except requests.HTTPError as http_err:
            self.logger.error(f"MsgID={msg_id}. [LINKER RETRIEVAL] HTTP error occurred: {http_err}.") # Specific HTTP error handling
            return f"[LINKER RETRIEVAL] HTTP error occurred: {http_err}"  
        except Exception as e:
            self.logger.error(f"MsgID={msg_id}. [LINKER RETRIEVAL] Error occurred during Sefaria Linker Querying: {e}.") # General error handling
            return f"[LINKER RETRIEVAL] Error occurred during Sefaria Linker Querying: {e}"

    def retrieve_docs_linker(self, screen_res: str, enriched_query: str, msg_id: str = '', filter_mode: str = 'primary'):
        '''
        Retrieves documents from the Sefaria Linker API based on a given query and screen_res query, applying filters to distinguish between primary and secondary sources.

        This function performs API calls to retrieve data relevant to enriched queries and processes that data based on specified filtering criteria.
        It is designed to support dynamic filtering of documents into primary or secondary categories, allowing for customized handling of search results.
        
        Parameters:
        screen_res (str): The screen_res query, which is used as part of the query to the Sefaria Linker.
        enriched_query (str): The enriched query string used to retrieve documents.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        filter_mode (str): Mode for filtering search results; valid options are 'primary' or 'secondary'. Defaults to 'primary'.
        
        Returns:
        list: A list of dictionaries, each representing a document that matches the search criteria. Each dictionary includes document details and an adjusted page rank.
        
        Raises:
        ValueError: If the provided filter_mode is not recognized, an error is raised indicating an invalid filter mode.

        Example:
        primary_results_linker = vh.retrieve_docs_linker(screen_res, enriched_query , msgid, 'primary')
        '''
        # Making a call to sefaria linker api
        json_input = self.query_sefaria_linker(text_title=screen_res, text_body=enriched_query, msg_id=msg_id)

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
                                sub_value['page_rank'] = 1e3
                                results.append(sub_value)
                    elif isinstance(value, (dict, list)):
                        # Continue search in deeper levels
                        traverse(value)
            elif isinstance(json_data, list):
                for item in json_data:
                    traverse(item)

        traverse(json_input)
        self.logger.info(f"MsgID={msg_id}. [LINKER RETRIEVAL] Sefaria linker document retrieval results: {results}")
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

        Example:
        p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict = vh.merge_linker_refs(retrieved_docs, p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict, msgid)
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
                
                self.logger.info(f"MsgID={msg_id}. [LINKER UPDATE SUCCESSFUL] Necessary fields are satisfied for this reference: ----new_reference_part: {new_reference_part} ----pr_score: {pr_score} ----new_category: {new_category} ----new_text: {new_text}")
            else:
                self.logger.info(f"MsgID={msg_id}. [LINKER UPDATE FAILED] Necessary fields are empty for this reference: ----new_reference_part: {new_reference_part} ----pr_score: {pr_score} ----new_category: {new_category} ----new_text: {new_text}")
                
        #sorting it by page rank score
        p_sorted_src_rel_dict = dict(sorted(p_sorted_src_rel_dict.items(), key=lambda item: item[1], reverse=True))
        self.logger.info(f"MsgID={msg_id}. [FINAL LINKER REFERENCE MERGE OUTPUT] ----p_sorted_src_rel_dict: {p_sorted_src_rel_dict} ----p_src_data_dict: {p_src_data_dict} ----p_src_ref_dict: {p_src_ref_dict}")

        return p_sorted_src_rel_dict, p_src_data_dict, p_src_ref_dict
        
    def topic_ontology(self, extraction: str = '', msgid: str = '', slugs_mode: bool = False):
        '''
        Processes and retrieves topic ontology data, either from a cache or by fetching new data if the cache is expired.

        This function checks for a cached file of all topics, loads it if it is still valid, or fetches and caches the data if the cache has expired.
        It also processes the extraction string to get topic names, retrieves corresponding topic slugs, and optionally fetches topic descriptions.
        The function ensures efficient access through caching and offers the option to return either slugs or descriptions for the topics.

        Parameters:
        extraction : str, optional
            A comma-separated string of topic names to process and search for, by default ''.
        msgid : str, optional
            Identifier for logging purposes, by default ''.
        slugs_mode : bool, optional
            Flag to determine if the function should return slugs instead of descriptions, by default False.

        Returns:
        list or dict
            If slugs_mode is True, returns a list of topic slugs.
            Otherwise, returns a dictionary with slugs as keys and topic descriptions as values.

        Example:
        topic_ont_dict = vh.topic_ontology(expanded_extraction, msgid)
        '''

        self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Starting topic ontology process.")
        cache_file = 'all_topics.json'
        
        def get_all_topics():
            cache_expiry = timedelta(days=1)
            topics = []  # Ensure topics is always initialized

            try:
                if os.path.exists(cache_file):
                    cache_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
                    if datetime.now() - cache_mtime < cache_expiry:
                        with open(cache_file, 'r') as file:
                            topics = json.load(file)
                        self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Loaded topics from cache.")
                    else:
                        topics = fetch_and_cache_topics()
                else:
                    topics = fetch_and_cache_topics()
            except Exception as e:
                self.logger.error(f"MsgID={msgid}. [ONTOLOGY] Exception occurred: {str(e)}")
            return topics

        def fetch_and_cache_topics():
            try:
                response = requests.get('https://www.sefaria.org/api/topics?limit=0', headers={"accept": "application/json"})
                if response.status_code == 200:
                    topics = response.json()
                    with open(cache_file, 'w') as file:
                        json.dump(topics, file)
                    self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Fetched and cached topics from Sefaria API.")
                else:
                    self.logger.error(f"MsgID={msgid}. [ONTOLOGY] Failed to fetch topics from Sefaria API.")
                    raise Exception(f"MsgID={msgid}. Failed to fetch topics from Sefaria API")
            except Exception as e:
                self.logger.error(f"MsgID={msgid}. [ONTOLOGY] Exception occurred while fetching topics: {str(e)}")
                topics = []
            return topics

        def preprocess_topic_names(extraction):
            topic_names = extraction.split(",")
            updated_topic_names = []
            for topic in topic_names:
                updated_name = topic.strip()
                updated_topic_names.append(updated_name)
                if updated_name.lower().startswith(('rabbi', 'rebbe')):
                    alt_name = updated_name[6:].strip()
                    if alt_name:
                        updated_topic_names.append(alt_name)
            return updated_topic_names
                    
        def find_topic_slugs(topic_names, all_topics):
            slugs = []
            name_set = {name.lower() for name in topic_names}
            for topic in all_topics:
                for title in topic.get('titles', []):
                    if title.get('text', '').lower() in name_set:
                        slugs.append(topic.get('slug', ''))
                        break
            self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Found topic slugs: {slugs}")
            return slugs

        def get_topic_descriptions(topic_slugs):
            descriptions = {}
            for slug in topic_slugs:
                response = requests.get(f'https://www.sefaria.org/api/v2/topics/{slug}')
                if response.status_code == 200:
                    topic_data = response.json()
                    if 'description' in topic_data and 'en' in topic_data['description']:
                        descriptions[slug] = topic_data['description']['en']
            self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Retrieved topic descriptions: {descriptions}")
            return descriptions
        
        # Process the extraction string
        topic_names = preprocess_topic_names(extraction)
        self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Extracted topic names: {topic_names}")

        # Get all topics
        all_topics = get_all_topics()

        # Find slugs for the topic names
        topic_slugs = find_topic_slugs(topic_names, all_topics)

        if slugs_mode:
            return topic_slugs
        else:        
            # Get descriptions for the topic slugs
            descriptions = get_topic_descriptions(topic_slugs)

            # Create a dictionary with topic names as keys and their descriptions as values
            final_descriptions = {}
            for slug, description in descriptions.items():
                desc = description.strip()
                if desc:
                    final_descriptions[slug] = desc

            self.logger.info(f"MsgID={msgid}. [ONTOLOGY] Final topic descriptions: {final_descriptions}")
            return final_descriptions

    def graph_traversal_retriever(self,
                                  screen_res: str,
                                  scripture_query: str,
                                  enriched_query: str,
                                  filter_mode_nodes: str | None = None,
                                  linker_results: list[dict]|None = None,
                                  semantic_search_results: list[tuple[Document, float]]|None = None,
                                  msg_id: str = ''):
        '''
        Find seed chunks based on linker results or semantic similarity, then traverse the graph to find related chunks in the local neighborhood.

        This function first identifies seed chunks using linker results or semantic search, and then traverses the graph to find related chunks within the neighborhood. Results are ranked based on relevance.

        Parameters:
        screen_res : dict
            The screen_res query, which is used as part of the query to the Sefaria Linker.
        scripture_query : str
            The query used to retrieve documents from the vector database.
        enriched_query : str
            The query enriched with additional context.
        filter_mode_nodes : str, optional
            Filter mode for 'primary' or 'secondary' references, optional.
        linker_results : list
            The results from the Sefaria Linker.
        semantic_search_results : list
            The results from the semantic search.
        msg_id : str, optional
            Identifier for logging purposes, by default ''.

        Returns:
        list
            A list of sorted chunks, ranked by relevance in descending order.

        Example:
        sel_p_retrieval_res, tok_count = vh.graph_traversal_retriever(
            screen_res=screen_res,
            scripture_query=scripture_query,
            enriched_query=enriched_query,
            linker_results=retrieval_res_linker,
            filter_mode_nodes=None,
            msg_id=msgid
        )
        '''

        # get seed chunks
        self.logger.info(f"MsgID={msg_id}. [GRAPH TRAVERSAL] Starting graph_traversal_retriever.")
        total_token_count = 0
        collected_chunks = []
        ranking_scores_collected_chunks = []
        if linker_results:
            if semantic_search_results:
                self.logger.warning(f"MsgID={msg_id}. [GRAPH TRAVERSAL] Both linker results and semantic search results are provided. Using linker results as seeds.")
            seed_chunks = self.get_linker_seed_chunks(linker_results=linker_results, msg_id=msg_id)
        elif semantic_search_results:
            seed_chunks_vector_db = [doc for doc, _ in semantic_search_results]
            seed_chunks = self.get_chunks_corresponding_to_nodes(seed_chunks_vector_db, msg_id=msg_id)
        else:
            raise ValueError(f"MsgID={msg_id}. [GRAPH TRAVERSAL] One of linker results or semantic search results need to be provided.")
        # rank seed chunks
        seed_chunks, token_count = self.select_reference(enriched_query, seed_chunks, msg_id=msg_id)
        total_token_count += token_count
        candidate_chunks, candidate_rankings, token_count = self.rank_documents(
            seed_chunks,
            enriched_query=enriched_query,
            scripture_query=scripture_query,
            msg_id=msg_id
        )
        total_token_count += token_count

        n_accepted_chunks = 0
        n_iter = 0 
        seed_iteration = True
        while n_accepted_chunks < self.config["database"]["kg"]["max_depth"]:          
            if len(candidate_chunks) == 0:
                break
            # Get the top chunk
            top_chunk = candidate_chunks.pop(0) 
            if not seed_iteration:
                collected_chunks.append(top_chunk)
                local_top_score = candidate_rankings.pop(0)
                ranking_scores_collected_chunks.append(local_top_score)
                n_accepted_chunks += 1
            # avoid final loop execution which does not add a chunk to collected_chunks anyways
            if n_accepted_chunks >= self.config["database"]["kg"]["max_depth"]:
                break
            else:
                n_iter +=1
            self.logger.info(f"MsgID={msg_id}. [GRAPH TRAVERSAL] Graph traversal iteration {n_iter} starts.")
            # Get the top node and neighbor nodes
            top_node = self.get_node_corresponding_to_chunk(top_chunk, msg_id=msg_id)
            neighbor_nodes_scores: list[tuple[Document, int]] = self.get_retrieval_results_knowledge_graph(
                url=top_node.metadata["url"],
                direction=self.config["database"]["kg"]["direction"],
                order=self.config["database"]["kg"]["order"],
                filter_mode_nodes=filter_mode_nodes,
                score_central_node=6.0,
                msg_id=msg_id
            )
            # Limit the amount of neighbors to top 15
            neighbor_nodes = [node for node, _ in neighbor_nodes_scores][:15]
            if not neighbor_nodes: 
                break
            candidate_chunks = self.get_chunks_corresponding_to_nodes(neighbor_nodes, msg_id=msg_id)
            # avoid re-adding the top chunk
            candidate_chunks = [chunk for chunk in candidate_chunks if chunk not in collected_chunks]
            candidate_chunks, token_count = self.select_reference(enriched_query, candidate_chunks, msg_id=msg_id)
            total_token_count += token_count
            candidate_chunks, candidate_rankings,  token_count = self.rank_documents(
                candidate_chunks,
                enriched_query=enriched_query,
                scripture_query=scripture_query,
                msg_id=msg_id
            )
            total_token_count += token_count
            seed_iteration = False
        retrieval_res_kg = sorted(zip(collected_chunks, ranking_scores_collected_chunks), key=lambda pair: pair[1], reverse=True)

        return retrieval_res_kg,  total_token_count

    def get_linker_seed_chunks(self, linker_results: list[dict],
                        filter_mode: str="primary", msg_id: str = '') -> list[Document]:
        '''
        Given linker results, get the corresponding seed chunks.

        This function first retrieves the seed nodes based on the linker results, and then finds the chunks corresponding to those seed nodes. 
        There is a one-to-many relationship between nodes and chunks.

        Parameters:
        linker_results : list
            Results from the linker API.
        scripture_query : str
            The scripture query to retrieve corresponding documents.
        msg_id : str, optional
            Message ID for logging purposes, particularly for Slack, by default ''.

        Returns:
        list
            A list of seed chunks.

        Example:
        seed_chunks = vh.get_linker_seed_chunks(linker_results=linker_results, msg_id=msg_id)
        '''

        self.logger.info(f"MsgID={msg_id}. [LINKER SEED CHUNKS] Starting get_linker_seed_chunks for KG search.")
        seeds: list[Document] = self.retrieve_nodes_matching_linker_results(linker_results, msg_id, filter_mode=filter_mode)
        seed_chunks: list[Document] = self.get_chunks_corresponding_to_nodes(seeds, msg_id=msg_id)
        return seed_chunks

    def rank_documents(self, chunks: list[Document], enriched_query: str, scripture_query: str|None=None, semantic_similarity_scores: list[float]|None = None,
                              filter_mode: str|None = None, msg_id: str = '') -> tuple[list[Document], list[float], int]:
        '''
        Rank the document candidates in descending order based on their relevance to the query.

        This function ranks the provided chunks (documents) based on their relevance to the query and returns a new list without modifying the input list.

        Parameters:
        chunks : list
            Langchain documents.
        enriched_query : str
            The query enriched with additional context.
        scripture_query : str
            The query used to retrieve documents from the vector database.
        semantic_similarity_scores : list, optional
            Pre-computed semantic similarity scores to save computational costs, if available.
        filter_mode : str, optional
            Specifies whether the references are 'primary' or 'secondary'. Set the mode if all documents are of the same type; set to None for mixed types.
            If set to 'secondary', no page rank scores are computed.

        Returns:
        tuple
            A tuple containing ranked chunks, ranking scores, and the total token count.

        Example:
        sorted_docs, sorted_ranking_scores, token_count = vh.rank_documents(
            documents=chunks,
            enriched_query=enriched_query,
            scripture_query=scripture_query,
            semantic_similarity_scores=semantic_similarity_scores,
            filter_mode=filter_mode,
            msg_id=msg_id
        )
        '''

        self.logger.info(f"MsgID={msg_id}. [RERANKING] Starting reranking chunks.")
        total_token_count = 0
        if not semantic_similarity_scores:
            if not enriched_query:
                raise ValueError(f"MsgID={msg_id}. Either provide semantic similarity scores or enriched query.")
            semantic_similarity_scores: np.array = self.compute_semantic_similarity_documents_query(chunks, query=enriched_query, msg_id=msg_id)
        reference_classes, token_count = self.get_reference_class(chunks, scripture_query=scripture_query, enriched_query=enriched_query, msg_id=msg_id)
        total_token_count += token_count

        if filter_mode == "secondary":
            page_rank_scores = np.ones((len(chunks), 1), dtype=float)
        else:
            page_rank_scores: np.array = self.get_page_rank_scores(chunks, msg_id=msg_id)

        # Combine the scores
        final_ranking_score = semantic_similarity_scores * reference_classes * page_rank_scores
        sort_indices = np.argsort(final_ranking_score, axis=0)[::-1].reshape(-1)
        ranking_scores = np.sort(final_ranking_score, axis=0)[::-1].reshape(-1).tolist()
        sorted_chunks = [chunks[i] for i in sort_indices]
        self.logger.info(f"MsgID={msg_id}. [RERANKING] sorted_chunks={[chunk.metadata['source'] for chunk in sorted_chunks]}, ranking_scores={ranking_scores}")
        return sorted_chunks, ranking_scores, total_token_count

    def compute_semantic_similarity_documents_query(self, documents: list[Document], query: str, msg_id: str = '') -> np.array:
        '''
        Compute the semantic similarity between a document and a query.

        This function calculates the semantic similarity score between the provided documents and the query.

        Parameters:
        documents : list
            Langchain documents to compare.
        query : str
            The query string against which the documents will be compared.

        Returns:
        float
            The similarity score between the documents and the query.

        Example:
        semantic_similarity_scores = vh.compute_semantic_similarity_documents_query(
            documents=documents, 
            query=enriched_query, 
            msg_id=msg_id
        )
        '''

        query_embedding = np.array(self.neo4j_vector.embedding.embed_query(text=query)).reshape(1, -1)
        document_embeddings = np.array([doc.metadata["embedding"] for doc in documents])
        if self.neo4j_vector._distance_strategy.value.lower() == "cosine":
            similarity = cosine_similarity(query_embedding, document_embeddings)
            relevance_score_function = self.neo4j_vector._select_relevance_score_fn()
            return relevance_score_function(similarity).reshape(-1, 1)
        else:
            raise NotImplementedError(f"MsgID={msg_id}. Distance strategy {self.neo4j_vector._distance_strategy.value} not implemented.")

    def get_reference_class(self, documents: list[Document], scripture_query: str, enriched_query: str, msg_id: str = '') -> np.array:
        '''
        Get the reference class for each document based on the query.

        This function determines the reference class for each document by analyzing how well they match the scripture and enriched queries.

        Parameters:
        documents : list
            Langchain documents to classify.
        scripture_query : str
            The query string used for retrieving documents from the vector database.
        enriched_query : str
            The query enriched with additional context.

        Returns:
        list
            An array of reference classes corresponding to each document.

        Example:
        reference_classes, token_count = vh.get_reference_class(
            documents=documents, 
            scripture_query=scripture_query, 
            enriched_query=enriched_query, 
            msg_id=msg_id
        )
        '''

        reference_classes = []
        total_token_count = 0
        for doc in documents:
            ref_data = doc.page_content + "... --Origin of this " + doc.metadata["source"]
            query = scripture_query if self.is_primary_document(doc) else enriched_query
            ref_class, token_count = self.classification(query=query, ref_data=ref_data, msg_id=msg_id)
            total_token_count += token_count
            reference_classes.append(ref_class)
        return np.array(reference_classes).reshape(-1, 1), total_token_count

    def get_page_rank_scores(self, documents: list[Document], msg_id: str = '') -> np.array:
        '''
        Get the PageRank scores for each document.

        This function retrieves the PageRank scores for the provided documents from their metadata, and performs batch-wise min-max scaling to normalize the scores.

        Parameters:
        documents : list
            Langchain documents for which to compute PageRank scores.
        msg_id : str, optional
            Message ID for logging purposes, by default ''.

        Returns:
        np.array
            An array of scaled PageRank scores.

        Example:
        page_rank_scores = vh.get_page_rank_scores(
            documents=documents, 
            msg_id=msg_id
        )
        '''
        page_rank_scores_raw = []
        for doc in documents:
            page_rank_score = doc.metadata["pagerank"]
            page_rank_scores_raw.append(page_rank_score)
        self.logger.info(f"MsgID={msg_id}. [PAGERANK] Retrieved raw pagerank scores={page_rank_scores_raw}")
        
        page_rank_scores_scaled = min_max_scaling(page_rank_scores_raw)
        self.logger.info(f"MsgID={msg_id}. [PAGERANK] Scaled pagerank scores={page_rank_scores_scaled}")
        return np.array(page_rank_scores_scaled).reshape(-1, 1)

    def is_primary_document(self, doc: Document) -> bool:
        '''
        Check if a document is a primary document.

        This function checks if the given document is considered a primary document by matching its source metadata against a predefined list of primary sources.

        Parameters:
        doc : Document
            The Langchain document to be checked.

        Returns:
        bool
            True if the document is a primary document, False otherwise.

        Example:
        res = vh.is_primary_document(doc)
        '''
        return any(s in doc.metadata['source'] for s in self.primary_source_filter)

    def get_chunks_corresponding_to_nodes(self, nodes: list[Document], batch_size: int = 20, max_nodes: int|None = None, unique_url: bool = True, msg_id: str = '') -> list[Document]:
        '''
        Given a list of nodes, return the chunks corresponding to each node.

        This function retrieves the chunks that correspond to a given list of nodes, with options to limit the number of nodes and batch size to avoid memory issues, and to ensure each node has a unique URL.

        Parameters:
        node : list
            The IDs of the nodes to retrieve corresponding chunks for.
        batch_size : int
            The number of documents to retrieve per query to avoid memory issues.
        max_nodes : int
            The maximum number of nodes to process.
        unique_url : bool
            Flag to determine whether to filter nodes such that each has a unique URL.

        Returns:
        list
            The IDs of the chunks corresponding to the nodes.

        Example:
        seed_chunks = self.get_chunks_corresponding_to_nodes(
            nodes=seed_chunks_vector_db, 
            msg_id=msg_id
        )
        '''

        if unique_url:
            seen_urls = set()
            nodes = [node for node in nodes if node.metadata["url"] not in seen_urls and not seen_urls.add(node.metadata["url"])]
        query_parameters = [
            {"versionTitle": node.metadata["versionTitle"], "url": node.metadata["url"]}
            for node in nodes[:max_nodes]
        ]
        self.logger.info(f"MsgID={msg_id}. [NODE2CHUNK] Using the following nodes to find corresponding chunks: {query_parameters}")
        query_string = """
        UNWIND $params AS param
        MATCH (n:Chunk)
        WHERE n.versionTitle = param.versionTitle AND n.url = param.url
        RETURN n
        """
        vector_records = []
        for i in range(0, len(query_parameters), batch_size):
            try:
                vector_records_batch = self.neo4j_vector.query(query_string, params={"params": query_parameters[i:i+batch_size]})
            except neo4j.exceptions.ServiceUnavailable:
                self.logger.warning(f"MsgID={msg_id}. Neo4j database is unavailable. Retrying.")
                sleep(1)
                vector_records_batch = self.neo4j_vector.query(query_string, params={"params": query_parameters[i:i+batch_size]})
            except BufferError:
                self.logger.warning(f"MsgID={msg_id}. Neo4j encountered an error. Retrying.")
                sleep(1)
                vector_records_batch = self.neo4j_vector.query(query_string, params={"params": query_parameters[i:i+batch_size]})
            vector_records += vector_records_batch
        self.logger.info(f"MsgID={msg_id}. [NODE2CHUNK] Found {len(vector_records)} node-corresponding chunks")
        return [convert_vector_db_record_to_doc(record) for record in vector_records]

    def get_node_corresponding_to_chunk(self, chunk: Document, msg_id: str = '') -> Document:
        '''
        Given a chunk, return the node corresponding to that chunk.

        This function retrieves the node that corresponds to a given chunk, represented as a document.

        Parameters:
        chunk : Document
            The document representing the chunk.

        Returns:
        Document
            The document representing the node corresponding to the chunk.

        Example:
        node = vh.get_node_corresponding_to_chunk(
            chunk=chunk, 
            msg_id=msg_id
        )
        '''
        query_parameters = {"url": chunk.metadata["url"], "versionTitle": chunk.metadata["versionTitle"]}
        self.logger.info(f"MsgID={msg_id}. [CHUNK2NODE] Using the following chunk to find a corresponding node: {query_parameters}")
        query_string="""
        MATCH (n:Records)
        WHERE n.url=$url
        AND n.versionTitle=$versionTitle
        RETURN n
        """
        with neo4j.GraphDatabase.driver(self.config["database"]["kg"]["url"], auth=(self.config["database"]["kg"]["username"], self.config["database"]["kg"]["password"])) as driver:
            nodes, _, _ = driver.execute_query(
            query_string,
            parameters_=query_parameters,
            database_=self.config["database"]["kg"] ["name"],)
        assert len(nodes) == 1
        node = nodes[0]
        self.logger.info(f"MsgID={msg_id}. [CHUNK2NODE] Found chunk-corresponding node for {query_parameters}")
        return convert_node_to_doc(node)

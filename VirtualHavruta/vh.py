# Load basic libraries
from __future__ import annotations
import operator
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from time import sleep
import os
import json
import yaml
import uuid6
import hdate
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
    min_max_scaling, part_res, find_matched_filters, construct_db_filter, load_selected_keys, merge_topics


# Main Virtual Havruta functionalities
# This class functions as a kind of utility class.  It takes in secrets, initializes connections, and provides utility functions.
# Long term, it's unclear if this class will be necessary, but for now, it remains.
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

        # Retrieve model and DB configs
        self.model_api = self.config['openai_model_api']
        self.chain_setups = self.config['llm_chain_setups']
        self.config_emb_db = self.config['database']['embed']
        self.config_kg_db = self.config['database']['kg']

        # Initialize Neo4j vector index 
        self.neo4j_vector = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(model=self.model_api['embedding_model']),
            index_name="index",
            url=self.config_emb_db['url'],
            username=self.config_emb_db['username'],
            password=self.config_emb_db['password'],
        )
        self.top_k = self.config_emb_db['top_k']

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
        self.neo4j_deeplink = self.config_kg_db['neo4j_deeplink']

        # Initialize prompt templates and LLM instances
        self.initialize_prompt_templates()
        self.initialize_llm_instances()

        self.metadata_ranges = load_selected_keys('./data/metadata_ranges.json', self.config["database"]["embed"]["metadata_fields"])
        self.topic_ranges = load_selected_keys('./data/metadata_ranges.json', self.config["database"]["embed"]["topic_fields"])

    def new_study_session(self, msgid: str = None, chat_callback: Optional[Callable] = None) -> StudySession:
        return StudySession(self, msgid, chat_callback=chat_callback)

    def initialize_prompt_templates(self):
        '''
        Initializes prompt templates for various chat interactions and updates class attributes accordingly.

        This function initializes prompt templates for different chat interaction categories such as anti-attack, adaptor, editor, optimization, and classification.
        It iterates over a list of categories, creates a prompt template for each category based on the 'system' template, and updates the class attributes with the generated templates.
        Additionally, it creates a separate prompt template for the QA (question-answering) category, including reference data.

        '''
        no_ref_categories = self.chain_setups['no_ref_chains']
        ref_categories = self.chain_setups['ref_chains']
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
        # Adding a condition to include json kwargs for models ending with '_json'
        for model_name, suffixes in self.chain_setups.items():
            if model_name.startswith(('main', 'support')):
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

    #todo: deprecate in favor of StudySession.make_prediction method
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

    def retrieve_docs_unfiltered(self, query: str):
        '''
        Retrieves documents that match a specified query and filters them based on whether they are primary or secondary sources, using a similarity search.

        This function performs a similarity search based on the provided query and retrieves documents that either match the characteristics of primary or secondary sources as defined by a filter set.
        The results are filtered by checking each document's metadata against a predefined set of source filters.
        The function logs the process to ensure transparency and is equipped to handle errors related to invalid filter modes, raising a ValueError if necessary.

        Parameters:
        query (str): The query string used to search for relevant documents.

        Returns:
        retrieval_res: Two list of documents - primary and secondary sources.

        Example:
        primary_retrieval_result = vh.retrieve_docs(query, msgid, 'primary')
        '''
        return self.neo4j_vector.similarity_search_with_relevance_scores(query, self.top_k)

    def retrieve_docs_metadata_filtering(self, query: str, metadata_filter: dict | None=None):
        '''
        Retrieves documents that match a specified query and filters them based on their metadata, using a similarity search.

        This function performs a similarity search based on the provided query and retrieves documents that match the metadata conditions as defined by a metadata_filter.
        The results are filtered by applying the metadata filters during semantic search.
        The function logs the process to ensure transparency.
        
        Parameters:
        query (str): The query string used to search for relevant documents.
        metadata_filter (dict): The metadata filter dictionary used to filter the search results during semantic search.
        
        Returns:
        list: A list of documents that meet the criteria of the specified metadata filter.

        Example:
        p_retrieval_res = vh.retrieve_docs_metadata_filtering(query, msgid, metadata_filter)
        '''
        # Convert primary_source_filter to a set for efficient lookup
        retrieved_res = self.neo4j_vector.similarity_search_with_relevance_scores(
            query, self.top_k, filter=metadata_filter
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
        self.logger.info(f"MsgID={msg_id}. [LINKER-GRAGH RETRIEVAL] Graph nodes retrieved using linker URLs: {['URL='+url+' SOURCE='+node.metadata['source'] for url, node in url_to_node.items()]}")
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
            direction=self.config_kg_db["direction"],
            order=self.config_kg_db["order"],
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
            with neo4j.GraphDatabase.driver(self.config_kg_db["url"], auth=(self.config_kg_db["username"], self.config_kg_db["password"])) as driver:
                neighbor_nodes, _, _ = driver.execute_query(
                query,
                parameters_=query_params,
                database_=self.config_kg_db["name"],)
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
        with neo4j.GraphDatabase.driver(self.config_kg_db["url"], auth=(self.config_kg_db["username"], self.config_kg_db["password"])) as driver:
            nodes, _, _ = driver.execute_query(
            query_string,
            parameters_=query_parameters,
            database_=self.config_kg_db["name"],)
        return [convert_node_to_doc(node) for node in nodes]

    #todo: deprecate in favor of StudySession?  Still used in graph_traversal_retriever
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

    #todo: deprecated in favor of StudySession
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

    #todo: deprecated in favor of StudySession
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

    def query_sefaria_linker(self, text_title="", text_body="", with_text=1, debug=0, max_segments=20, msg_id: str = ''):
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
            raise ValueError(f"Invalid filter_mode: {filter_mode}")

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
        while n_accepted_chunks < self.config_kg_db["max_depth"]:
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
            if n_accepted_chunks >= self.config_kg_db["max_depth"]:
                break
            else:
                n_iter +=1
            self.logger.info(f"MsgID={msg_id}. [GRAPH TRAVERSAL] Graph traversal iteration {n_iter} starts.")
            # Get the top node and neighbor nodes
            top_node = self.get_node_corresponding_to_chunk(top_chunk, msg_id=msg_id)
            neighbor_nodes_scores: list[tuple[Document, int]] = self.get_retrieval_results_knowledge_graph(
                url=top_node.metadata["url"],
                direction=self.config_kg_db["direction"],
                order=self.config_kg_db["order"],
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

    # todo: deprecated in favor or StudySession.  Still used by Graph Traversal Retriever
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

    #todo deprecate in favor of SourceCollection
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

    #todo: deprecate in favor of SourceCollection
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
        with neo4j.GraphDatabase.driver(self.config_kg_db["url"], auth=(self.config_kg_db["username"], self.config_kg_db["password"])) as driver:
            nodes, _, _ = driver.execute_query(
            query_string,
            parameters_=query_parameters,
            database_=self.config_kg_db["name"],)
        assert len(nodes) == 1
        node = nodes[0]
        self.logger.info(f"MsgID={msg_id}. [CHUNK2NODE] Found chunk-corresponding node for {query_parameters}")
        return convert_node_to_doc(node)


class StudySession:
    def __init__(self, vh: VirtualHavruta, msgid: str = None, chat_callback: Optional[Callable] = None):
        self.session_id = msgid or uuid6.uuid7().hex
        self.chat = chat_callback or (lambda x: None)
        self.graph_flag = None
        self.main_response = None
        self.topic_ont_dict = None
        self.matched_topics_in_query = None
        self.expanded_extraction = None
        self.infer_topics_flag = False
        self.topic_slugs = []
        self.situational_info = None
        self.debug_flag = False
        self.source_collection = None
        self.formatted_sources = None
        self.all_citations = None
        self.history = []  # Track all state changes
        self._current_state = {}
        self.vh = vh
        self.logger = vh.logger
        self._total_tokens = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._successful_requests = 0
        self._total_cost = 0.0
        self._request_count = 0

        self.original_query = None
        self.edited_query = None

        self.translation = None  # translated query
        self.extraction = None  # extracted key concepts
        self.elaboration = None  # elaboration details
        self.challenge = None  # challenges identified
        self.quotation = None  # any related quotations
        self.proposal = None  # potential directions proposed

        self.linker_query = None
        self.enriched_query = None
        self.scripture_query = None
        self.matched_filters = None

        self.primary_citations = None
        self.secondary_citations = None
        self.debug = {}

    # todo: classify costs by model used

    def mark_as_eval(self):
        self.session_id = "eval_" + self.session_id

    def _setup(self):
        self.retrieve_situational_info()
        self.ingest_query()
        self.set_derivative_queries()
        # todo: This takes a while and offers dubious value.  Refine.
        if self.infer_topics_flag:
            self.infer_topics()
        self.set_filters()
        self.find_explicit_citations()    # Retrieve primary and secondary documents from linker api

    def retrieve(self, query: str) -> SourceCollection:
        self.original_query = query
        self._setup()

        # The source collection grabs session_id, matched_filters, and all_citations from self.
        # todo: should that be passed more explicitly?
        self.source_collection = SourceCollection(self, debug_flag=self.debug_flag)
        self.source_collection.retrieve_with_semantic_search()
        self.source_collection.rank()
        self.source_collection.merge()
        return self.source_collection

    def retrieve_and_generate(self, query: str):
        self.original_query = query
        self._setup()
        response = {}
        self.source_collection = SourceCollection(self, debug_flag=self.debug_flag)
        if self.graph_flag:
            self.chat("*I'm going to search for the answer in the knowledge graph.*")
            self.source_collection.retrieve_with_graph()
        # todo: fix this intrusion into source_collection
        if not self.source_collection.graph_retrieval_successful:
            self.chat("*I'm going to search for the answer in the database.*")
            self.source_collection.retrieve_with_semantic_search()
        self.source_collection.rank()
        self.source_collection.merge()
        citations_string = self.source_collection.as_citations_string()
        ref_data = self.source_collection.as_reference_data_string() + f"\nAd-hoc supplementary information: [Situational Context] {self.situational_info} [Tentative Translation in English] {self.translation} [Thoughts, Challenges, and Potential Ways to Answer] {self.elaboration} {self.challenge} {self.proposal} [Related Topics in Sefaria Database] {self.topic_ont_dict}"

        # Formulate the main response based on the combined reference data
        main_response = self.qa(self.edited_query, ref_data)

        if not self.source_collection.is_empty():
            if "@IRRELEVANT-SOURCE@" in main_response:
                final_msg = part_res(main_response, '@IRRELEVANT-SOURCE@')
                self.logger.warning(f"SessionID={self.session_id}. Model found irrelevant references!")
            else:
                final_msg = main_response + "\n"
        else:
            # If no citations are available, inform the user
            final_msg = "At the moment I couldn't find directly relevant information in my database. I appreciate your understanding."

        self.logger.info(f"SessionID={self.session_id}. [FINAL RESPONSE] {final_msg}")
        self.logger.info(f"SessionID={self.session_id}. {self.get_costs_as_string()}")
        response['answer'] = final_msg
        response["citations"] = citations_string
        if self.debug_flag:
            response['debug'] = self.debug
        return response

    def retrieve_situational_info(self):
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

        #todo: This may be off by a day if the server time and user timezone differ enough at the wrong time of day
        now = datetime.now()
        h = hdate.HDate(now, hebrew=False) # Includes day of week
        self.situational_info = f"[Situational Info] It is now {h}, ({now.strftime('%d/%m/%Y %H:%M:%S')}).  "

        #todo: This will retrieve the Israel Parasha when Israel and the disapora differ
        response = requests.get("https://www.sefaria.org/api/calendars", headers={"accept": "application/json"})
        if response.status_code == 200:
            data = response.json()
            parasha = data["calendar_items"][0]
            self.situational_info += f"The current Parasha is {parasha["displayValue"]["en"]} ({parasha["displayValue"]["he"]}), which is found in {parasha["ref"]}.  "

        self.logger.info(f"SessionID={self.session_id}. [SITUATIONAL INFO] Retrieved current situation: {self.situational_info}")

    def ingest_query(self):
        self.original_query = self.original_query.strip()
        self.logger.info(f"SessionID={self.session_id}. [INGESTION] Original query={self.original_query}")
        # Handle possible adversarial attacks by checking the query content
        detection, explanation = self.anti_attack(self.original_query)

        # If detected, adapt the query, otherwise, edit the query
        if "Y" in detection:
            self.edited_query = self.adaptor(self.original_query)
            self.logger.info(f"Caught attack. Adaptation result: {self.edited_query}")

            if not self.edited_query or '@CANNOT-ADAPT@' in self.edited_query:
                raise NoAnswerError("Adaptation failed")
            self.chat(f"*Umm...I'm reconsidering your question and for now I interpret it as: {self.edited_query}*")

        else:
            self.edited_query = self.editor(self.edited_query)

        return

    def set_derivative_queries(self):
        self.optimizer(self.edited_query)
        self.linker_query = f"{part_res(self.original_query)} {part_res(self.edited_query)}"
        self.enriched_query = f"{part_res(self.translation)} {part_res(self.extraction)} {part_res(self.elaboration)} {part_res(self.proposal)} {part_res(self.quotation)}"
        if self.quotation:
            self.scripture_query = f"{part_res(self.quotation)} {part_res(self.extraction)}"
        else:
            self.scripture_query = f"{part_res(self.translation)} {part_res(self.extraction)} {part_res(self.proposal)}"
        if self.debug:
            self.chat("*Well, these are not the final response but merely some of my preliminary and tentative thoughts while I go on considering your question:*")
            self.chat(part_res(self.elaboration))
            self.chat(part_res(self.challenge))
            self.chat(part_res(self.proposal))
            self.chat(f"*Here are quotes potentially related to your query:* {self.quotation}")
            self.chat( f"*And here are topics, authors, document categories, or era names potentially related to your query: {self.extraction}*")

    def infer_topics(self):
        self.topic_slugs = self.vh.topic_ontology(self.extraction, self.session_id, True)

        #todo: Are all of these needed at the class level?
        self.matched_topics_in_query = find_matched_filters(f"{self.edited_query} {self.translation}", self.vh.topic_ranges)
        self.expanded_extraction = merge_topics(self.extraction, self.matched_topics_in_query)
        self.logger.info(f"SessionID={self.session_id}. [TOPIC EXPANSION] Topics matched in query={self.matched_topics_in_query}. Extracted key concepts after merging={self.expanded_extraction}")

        self.topic_ont_dict = self.vh.topic_ontology(self.expanded_extraction, self.session_id)
        if not self.topic_ont_dict:
            self.logger.info(f"SessionID={self.session_id} [ONTOLOGY] No topics found for the given extraction.")
            if self.debug_flag:
                self.debug['found_topic_desc'] = False
        else:
            self.logger.info(f"SessionID={self.session_id} [ONTOLOGY] Topics found for the given extraction={self.topic_ont_dict}")
            if self.debug_flag:
                topic_ont_results_msg = ""
                # Iterate over the result dictionary and add as a string to post at the end
                self.debug['topics'] = []
                for topic, description in self.topic_ont_dict.items():
                    self.debug['topics'].append({
                        'topic': topic,
                        'description': description,
                    })

    def set_filters(self):
        self.matched_filters = find_matched_filters(f"{self.extraction} {', '.join(self.topic_slugs)}", self.vh.metadata_ranges)
        self.debug["matched_filters"] = self.matched_filters
        self.logger.info(f"SessionID={self.session_id}. [METADATA FILTERING] Filters matched in query={self.matched_filters}.")
        self.chat(f"*Here are the matched filters: {self.matched_filters}.")

    def has_filters(self):
        return bool(self.matched_filters)

    def find_explicit_citations(self):
        #todo: refactor into one linker call that gets parsed into two buckets
        self.primary_citations = self.vh.retrieve_docs_linker(self.linker_query, self.enriched_query, self.session_id, 'primary')
        self.secondary_citations = self.vh.retrieve_docs_linker(self.linker_query, self.enriched_query, self.session_id, 'secondary')
        self.all_citations = self.primary_citations + self.secondary_citations
        return

    def set_debug_flag(self, flag: bool):
        self.debug_flag = flag

    def set_graph_flag(self, flag: bool):
        self.graph_flag = flag

    def set_infer_topics_flag(self, flag: bool):
        self.infer_topics_flag = flag

    def update_costs(self, cb):
        """
        cb: OpenAi Callback object
        """
        self._total_tokens += cb.total_tokens
        self._prompt_tokens += cb.prompt_tokens
        self._completion_tokens += cb.completion_tokens
        self._successful_requests += cb.successful_requests
        self._total_cost += cb.total_cost
        self._request_count += 1

    def get_costs_as_string(self):
        return f"Total cost: {self._total_cost}, Total tokens: {self._total_tokens}, Prompt tokens: {self._prompt_tokens}, Completion tokens: {self._completion_tokens}, Successful requests: {self._successful_requests}, Request count: {self._request_count}"

    def make_prediction(self, chain, query: str, action: str, ref_data: str = ''):
        """
        Executes a prediction using a specified language model chain
        Logs actions and tracks costs
        Catches and logs any exceptions that occur during the prediction process, including token expenditure.

        Parameters:
        chain (LanguageModelChain): The specific language model chain used for prediction.
        query (str): The input query string for which the prediction is needed.
        action (str): The type of action the model is performing, used for logging.
        msg_id (str, optional): A message identifier used for logging purposes; defaults to an empty string.
        ref_data (str, optional): Additional reference data to be included in the prediction request; defaults to an empty string.

        Returns:
        str: the result of the prediction

        Example:
        make_prediction(self.chat_llm_chain_anti_attack, query, "ANTI-ATTACK", msg_id)
        """

        with get_openai_callback() as cb:
            try:
                res = chain.predict(human_input=query, ref_data=ref_data) if ref_data else chain.predict(
                    human_input=query)
                self.logger.info(
                    f"SessionID={self.session_id}. [INFERENCE] Spent {cb.total_tokens} tokens for {action}. Query={query}. Reference data={ref_data}. Result={res}.")
            except Exception as e:
                self.logger.error(
                    f"SessionID={self.session_id}. [INFERENCE] Spent {cb.total_tokens} tokens for {action} but failed. Error is {e}.")
                res = ''
            self.update_costs(cb)
            return res

    def anti_attack(self, query: Optional[str] = None):
        '''
        Analyzes a query for potential attacks using a language model chain specialized in anti-attack tasks, returning the detection status, explanation, and token count.

        This function submits a query to an anti-attack model, which assesses the text for elements that might constitute an attack or harmful content.
        The model's response is expected to include a detection status and an explanation, separated by a special delimiter.
        If the parsing of the response fails, the function logs the error and defaults the detection to 'N' (No) with an empty explanation.
        This ensures reliable operation even in cases of unexpected model output or processing errors.

        Parameters:
        query (str): The query string to be analyzed for potential attacks.

        Returns:
        tuple: A tuple containing the detection status (str), and an explanation (str).

        Raises:
        Exception: Catches and logs any exception that occurs during response parsing, setting default values for the detection status and explanation.

        Example:
        detection, explanation = vh.anti_attack(query)
        '''
        adv_res = self.make_prediction(self.vh.chat_llm_chain_anti_attack, query, "ANTI-ATTACK")
        try:
            detection, explanation = adv_res.split('@SEP@')
        except Exception as e:
            self.logger.error(
                f"SessionID={self.session_id}. [Anti-Attack] Error occurred during attack detection: {e}.")
            detection, explanation = 'N', ''
        return detection, explanation

    def adaptor(self, query: Optional[str] = None):
        '''
        Processes a query using a language model chain optimized for adaptation tasks, returning the adapted result along with the token count.

        This function sends a query to an adaptation-specific model, which modifies the query to fit particular contexts or requirements.
        It retrieves the adapted text and the number of tokens used in the model's response.
        The function is useful for tasks requiring contextual modifications or specific formatting.
        It also logs each transaction with an optional message identifier, aiding in monitoring and debugging processes.

        Parameters:
        query (str): The query string to be adapted by the model.

        Returns:
        tuple: A tuple containing the adapted text (str) and the token count (int) used in generating that adapted text.

        Example:
        screen_res = vh.adaptor(query)
        '''
        return self.make_prediction(self.vh.chat_llm_chain_adaptor, query, "ADAPTATION")

    def editor(self, query: Optional[str] = None):
        """
        Performs editing on a given query using a language model chain optimized for editing tasks, returning the edited result.
        This function sends a query to an editing-optimized model, which processes and refines the text to improve clarity, style, or correctness.
        The function returns the edited output.

        Parameters:
        query (str): The query string to be edited by the model.  If not present, use get_state("original_query")

        Returns:
        the edited text (str)

        Example:
        screen_res = vh.editor(query)
        """
        input_query = query or self.original_query
        return self.make_prediction(self.vh.chat_llm_chain_editor, input_query, "EDITING")

    def optimizer(self, query: Optional[str] = None):
        '''
        Optimizes a query using a chain of language models dedicated to prompt optimization, extracting various components from the optimization results.

        This function submits a query to an optimization model, which processes the query and returns structured optimization results.
        These results are expected to contain components such as translation, key concepts, elaboration, quotation, challenges, and potential directions.
        The function decodes the JSON response, extracts these components, and returns them along with the token count used in the operation.
        Errors during JSON processing are logged, and default values are used if an error occurs, ensuring the function remains robust across different scenarios.

        Parameters:
        query (str): The query string to be optimized by the model.

        Returns:
        tuple: A tuple containing the translated query, extracted key concepts, elaboration details, any related quotations, challenges identified, potential directions proposed

        Raises:
        Exception: Catches and logs any exception that occurs during the JSON parsing and sets all output components to empty strings as a fallback.

        Example:
        translation, extraction, elaboration, quotation, challenge, proposal, tok_count = vh.optimizer(screen_res, msgid)
        '''

        input_query = query or self.edited_query

        optimizer_result = self.make_prediction(self.vh.chat_llm_chain_optimization, input_query,
                                                 "PROMPT OPTIMIZATION")
        try:
            opt_res_json = json.loads(optimizer_result)
            self.translation = opt_res_json['Translation']
            self.extraction = opt_res_json['Key-Concepts']
            self.elaboration = opt_res_json['Elaboration']
            self.quotation = opt_res_json['Quotation']
            self.challenge = opt_res_json['Challenge']
            self.proposal = opt_res_json['Potential-Directions']

            #todo: This is repetative. Debug could be extracted from objects attrs when needed.
            if self.debug_flag:
                self.debug |= {
                    'elaboration': self.elaboration,
                    'challenge': self.challenge,
                    'proposal': self.proposal,
                    'quotes': self.quotation,
                    'extraction': self.extraction,
                }

        except Exception as e:
            self.logger.error(
                f"SessionID={self.session_id}. [OPTIMIZATION] Error occurred during PROMPT OPTIMIZATION: {e}.")
            self.translation = ""
            self.extraction = ""
            self.elaboration = ""
            self.quotation = ""
            self.challenge = ""
            self.proposal = ""
        return

    def qa(self, query: str, ref_data: str):
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

        self.main_response = self.make_prediction(self.vh.chat_llm_chain_qa, query, "qa", ref_data)
        return self.main_response


class NoAnswerError(Exception):
    pass


class RankedDocuments:
    def __init__(self, rel_dict: dict | None = None, data_dict: dict | None = None, ref_dict: dict | None = None):
        self.rel_dict = rel_dict or {}
        self.data_dict = data_dict or {}
        self.ref_dict = ref_dict or {}

    def union(self, other: RankedDocuments) -> RankedDocuments:
        return RankedDocuments(self.rel_dict | other.rel_dict, self.data_dict | other.data_dict, self.ref_dict | other.ref_dict)

    def __or__(self, other: RankedDocuments) -> RankedDocuments:
        return self.union(other)

    def first_n(self, n: int) -> list[tuple[str, float]]:
        """
        Returns the first n items in the relevance dictionary.
        """
        return list(self.rel_dict.items())[:n]

    def merge_linker_refs(self, retrieved_docs: list) -> RankedDocuments:
        """
        Returns a new RankedDocuments object resulting from the merging linker reference results into this object.

        Parameters:
            retrieved_docs (list): Contains dictionaries with page content and metadata including URL and text.

        Returns:
            RankedDocuments: A new RankedDocuments object with the merged results.
        """

        # iterating each document in reverse order
        for document in reversed(retrieved_docs):

            # Extract necessary data to be written and to be checked
            # Extracting the URL
            new_url = 'https://www.sefaria.org/' + document['url'] if document['url'] else None
            # Extracting the page_rank score for sorting
            pr_score = float(document['page_rank']) if document['page_rank'] else None
            # Extracting the Category
            new_category = document['primaryCategory'] if document['primaryCategory'] else None
            # Extracting the Reference Part
            new_reference_part = document['url'] if document['url'] else None
            new_ref = f"Reference: {new_reference_part}. Version Title: -, Document Category: {new_category}, URL: {new_url}"
            # Extracting the english text
            new_text = ' '.join(document['en']) if document['en'] else ""

            new_obj = RankedDocuments()
            # Update sorted source relevance dictionary if necessary fields are satisfied
            if new_reference_part and pr_score and new_category and new_text:
                # Commit changes if fields are satisfied

                new_obj.rel_dict = self.rel_dict | {new_url: pr_score}

                # Merges the existing dict over the new ref.  If the key already exists in the old one, append the new ref to the existing one.
                new_obj.ref_dict = {new_url: new_ref} | self.ref_dict
                if new_url in self.ref_dict and new_ref not in self.ref_dict[new_url]:
                    new_obj.ref_dict[new_url] += " | " + new_ref

                new_obj.data_dict = {new_url: new_text} | self.data_dict
                if new_url in self.data_dict and new_text not in self.data_dict[new_url]:
                    new_obj.data_dict[new_url] += "..." + new_text

                # todo: fix logger
                # self.logger.info(
                #    f"SessionID={self.session_id}. [LINKER UPDATE SUCCESSFUL] Necessary fields are satisfied for this reference: ----new_reference_part: {new_reference_part} ----pr_score: {pr_score} ----new_category: {new_category} ----new_text: {new_text}")
            # else:
                # self.logger.info(
                #    f"SessionID={self.session_id}. [LINKER UPDATE FAILED] Necessary fields are empty for this reference: ----new_reference_part: {new_reference_part} ----pr_score: {pr_score} ----new_category: {new_category} ----new_text: {new_text}")

        # sorting it by page rank score
        new_obj.rel_dict = dict(sorted(new_obj.rel_dict.items(), key=lambda item: item[1], reverse=True))
        # self.logger.info(
        #    f"SessionID={self.session_id}. [FINAL LINKER REFERENCE MERGE OUTPUT] ----p_sorted_src_rel_dict: {self.rel_dict} ----p_src_data_dict: {self.data_dict} ----p_src_ref_dict: {self.ref_dict}")

        return new_obj


class SourceCollection:
    def __init__(self, study_session: StudySession, debug_flag: bool = False):
        self.vh = study_session.vh
        self.session = study_session
        self.logger = self.session.logger
        self.session_id = self.session.session_id
        self.matched_filters = self.session.matched_filters
        self.all_citations = self.session.all_citations

        self.debug_flag = debug_flag
        self.debug = {}

        self.all_primary_documents = None
        self.selected_primary_docs = None
        self.ranked_primary_documents = RankedDocuments()
        self.selected_secondary_docs = None
        self.ranked_secondary_documents = RankedDocuments()

        self.all_ranked_docs = None
        self.final_reference_docs = None
        self.retrieval_set = None
        self.retrieval_is_filtered = None
        self.graph_retrieval_successful = False

    def retrieve_with_semantic_search(self):
        self.retrieval_is_filtered = False

        if self.has_filters():
            metadata_filter = construct_db_filter(self.matched_filters)
            self.debug["metadata_filter"] = metadata_filter
            self.logger.info(f"SessionID={self.session_id}. [RETRIEVAL] Metadata filtering at work. Retrieving references using this query: {self.session.scripture_query} and this metadata filter {metadata_filter}")
            self.retrieval_set = self.vh.retrieve_docs_metadata_filtering(self.session.scripture_query, metadata_filter)
            self.retrieval_is_filtered = bool(self.retrieval_set)


        # If no results are returned from semantic search with metadata filtering, do a simple semantic search
        if not self.retrieval_is_filtered:
            self.retrieval_set = self.vh.retrieve_docs_unfiltered(self.session.scripture_query)

        # todo: get this vh var local
        primary_predicate = lambda doc: any(s in doc[0].metadata['source'] for s in self.vh.primary_source_filter)

        primary_docs = [d for d in self.retrieval_set if primary_predicate(d)]
        self.selected_primary_docs = self.select_reference(primary_docs)

        if not self.retrieval_is_filtered:
            secondary_docs = [d for d in self.retrieval_set if not primary_predicate(d)]
            self.selected_secondary_docs = self.select_reference(secondary_docs)

    #todo: refactor graph_traversal_retriever
    def retrieve_with_graph(self):
        if self.all_citations:
            sel_p_retrieval_res, tok_count = self.vh.graph_traversal_retriever(
                screen_res=self.session.edited_query,
                scripture_query=self.session.scripture_query,
                enriched_query=self.session.enriched_query,
                linker_results=self.all_citations,
                filter_mode_nodes=None,
                msg_id=self.session_id
              )
            self.session._total_tokens += tok_count  # Update token count.  todo: Other numbers won't be consistent until refactor.
            self.graph_retrieval_successful = bool(sel_p_retrieval_res)
            if not self.graph_retrieval_successful and self.debug_flag:
                self.debug['graph_traversal_failed'] = True
        elif self.debug_flag:
            self.debug['graph_traversal_failed'] = True

    def rank(self):
        """
        Sort the selected primary and secondary documents based on their relevance to the query.
        """
        if self.graph_retrieval_successful:
            self.ranked_primary_documents = self.sort_reference(self.selected_primary_docs, None)

        else:
            if self.selected_primary_docs:
                self.ranked_primary_documents = self.sort_reference(self.selected_primary_docs, 'primary')

            if not self.retrieval_is_filtered and self.selected_secondary_docs:
                self.ranked_secondary_documents = self.sort_reference(self.selected_secondary_docs, 'secondary')

    def merge(self):
        self.all_primary_documents = self.ranked_primary_documents.merge_linker_refs(self.all_citations) \
            if self.has_citations() \
            else self.ranked_primary_documents

        self.final_reference_docs = self.all_primary_documents.first_n(self.vh.num_primary_citations) \
               + self.ranked_secondary_documents.first_n(self.vh.num_secondary_citations)
        self.all_ranked_docs = self.all_primary_documents | self.ranked_secondary_documents

    def is_empty(self):
        return not self.final_reference_docs

    def as_docs(self) -> list[tuple[str, float]]:
        """
        Returns the final reference documents as a list of tuples, where each tuple contains a document URL and its relevance score.

        :return: List of tuples, each with document URL and relevance score.
        """
        return self.final_reference_docs

    def as_citations_string(self):
        citation_parts = []
        for n, (k, rel_score) in enumerate(self.final_reference_docs, 1):
            citation_parts.append(f"\n{n}. {self.all_ranked_docs.ref_dict[k]}")
            self.logger.info(
                    f"SessionID={self.session_id}. [GENERATE REFERENCE STRING] Included this reference: {k}. Relevance score = {rel_score}."
                )

        return ''.join(citation_parts)

    def as_reference_data_string(self):
        ref_data_parts = []
        for n, (k, rel_score) in enumerate(self.final_reference_docs, 1):
            ref_data_parts.append(
                f"\n #Reference {n}# {self.all_ranked_docs.data_dict[k]}... --Origin of this {self.all_ranked_docs.ref_dict[k]} \n")
        return ''.join(ref_data_parts)


    def select_reference(self, retrieval_res):
        '''
        todo: rename to something like filter_references_with_llm?
        Based on the provided query and retrieval_res, select useful references using a chained language model, returning the selected retrieval_res and token count.

        This function selects retrieval results based on a language model specifically tuned for selection tasks.
        It captures the selected retrieval results, which are expected to be a list of documents, and the count of tokens used by the model.
        If the function's output cannot be converted to a list of documents due to an error, the function logs the error and defaults the selected results to [].
        This ensures robust error handling and maintains the integrity of the selection process under all conditions.

        Parameters:
        retrieval_res (list): A list of retrieved documents.

        Returns:
        list: The selected retrieval results (list of documents)

        Raises:
        Exception: Catches and logs any exception that occurs during the selection process, defaulting the result to [] and 0.

        Example:
        seed_chunks = vh.select_reference(enriched_query, seed_chunks, msg_id=msg_id)
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
            selected_idx = self.selector(self.session.enriched_query, conc_ref_data)
            selected_retrieval_res = [retrieval_res[i] for i in selected_idx]
        except Exception as e:
            self.logger.error(f"SessionID={self.session_id}. Reference selection result was set to []. Error message is {e}.")
            selected_retrieval_res = []
        return selected_retrieval_res


    def sort_reference(self, retrieval_res, filter_mode: str | None = 'primary') -> RankedDocuments:
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
        tuple: A tuple containing sorted source relevance dictionary, source data dictionary, source reference dictionary

        Notes:
        The function is robust to variations in data and manages complex scenarios where multiple references may have the same URL but different content or sources. It effectively manages and logs all operations to ensure data integrity and traceability.

        Example:

        '''

        documents, semantic_similarity_scores = zip(*retrieval_res)
        # todo: this next method is producing a cross-product
        sorted_docs, sorted_ranking_scores = self.rank_documents(
            documents,
            semantic_similarity_scores=semantic_similarity_scores,
            filter_mode=filter_mode
        )

        retrieval_res_ranked = list(zip(sorted_docs, sorted_ranking_scores))
        return self.merge_references_by_url(retrieval_res_ranked)

    def rank_documents(self, chunks: list[Document], semantic_similarity_scores: list[float]|None = None,
                              filter_mode: str|None = None) -> tuple[list[Document], list[float]]:
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
            A tuple containing ranked chunks, ranking scores

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

        self.logger.info(f"SessionID={self.session_id}. [RERANKING] Starting reranking chunks.")
        if semantic_similarity_scores:
            semantic_similarity_scores = np.array(semantic_similarity_scores).reshape((-1, 1))
        else:
            if not self.session.enriched_query:
                raise ValueError(f"SessionID={self.session_id}. Either provide semantic similarity scores or enriched query.")
            semantic_similarity_scores: np.array = self.vh.compute_semantic_similarity_documents_query(chunks, query=self.session.enriched_query, msg_id=self.session_id)

        reference_classes = self.get_reference_class(chunks)

        if filter_mode == "secondary":
            page_rank_scores = np.ones((len(chunks), 1), dtype=float)
        else:
            page_rank_scores: np.array = self.vh.get_page_rank_scores(chunks, msg_id=self.session_id)

        # Combine the scores
        final_ranking_score = semantic_similarity_scores * reference_classes * page_rank_scores
        sort_indices = np.argsort(final_ranking_score, axis=0)[::-1].reshape(-1)
        ranking_scores = np.sort(final_ranking_score, axis=0)[::-1].reshape(-1).tolist()
        sorted_chunks = [chunks[i] for i in sort_indices]
        self.logger.info(f"SessionID={self.session_id}. [RERANKING] sorted_chunks={[chunk.metadata['source'] for chunk in sorted_chunks]}, ranking_scores={ranking_scores}")
        return sorted_chunks, ranking_scores

    #todo: make this an instanciating method of RankedDocuments?
    def merge_references_by_url(self, retrieval_res: list[tuple[Document, float]]) -> RankedDocuments:
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
        self.logger.info(f"SessionID={self.session_id}. [MERGE REFERENCE] sorted_src_rel_dict={sorted_src_rel_dict}, src_data_dict={src_data_dict}, src_ref_dict={src_ref_dict}.")
        # Return the sorted source relevance dictionary, source data dictionary, source reference dictionary, and token count

        return RankedDocuments(sorted_src_rel_dict, src_data_dict, src_ref_dict)

    def selector(self, query: str, ref_data: str):
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

        response = self.session.make_prediction(self.vh.chat_llm_chain_selector, query, "SELECTOR", ref_data)
        try:
            if response.strip() == ',':
                selected_idx = []
            else:
                selected_idx = [int(x) for x in response.split(',') if x]
        except Exception as e:
            self.logger.error(
                f"SesssionID={self.session_id}. LLM SELECTOR result was set to []. Error message is {e}."
            )
            selected_idx = []

        return selected_idx

    def get_reference_class(self, documents: list[Document]) -> np.array:
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
        for doc in documents:
            ref_data = doc.page_content + "... --Origin of this " + doc.metadata["source"]
            query = self.session.scripture_query if self.is_primary_document(doc) else self.session.enriched_query
            ref_class = self.classification(query=query, ref_data=ref_data)
            reference_classes.append(ref_class)
        return np.array(reference_classes).reshape(-1, 1)

    def classification(self, query: str, ref_data: str):
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
        ref_class = self.session.make_prediction(self.vh.chat_llm_chain_classification, query, "CLASSIFICATION", ref_data)
        try:
            ref_class = int(ref_class)
        except Exception as e:
            self.logger.error(
                f"SessionID={self.session_id}. [CLASSIFICATION] Result was set to 0. Error message is {e}.")
            ref_class = 0
        return ref_class

    def has_filters(self):
        return bool(self.matched_filters)

    def has_citations(self):
        return bool(self.all_citations)

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
        return any(s in doc.metadata['source'] for s in self.vh.primary_source_filter)

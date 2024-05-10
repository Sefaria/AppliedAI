# Load basic libraries
import os
import yaml

# Import Virtual Havruta class
from VirtualHavruta import VirtualHavruta

# Import Slack SDK modules for bot interaction
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

# Utility function to create a logger for application logging
from VirtualHavruta.util import create_logger, part_res

# Load configurations for database, Slack, and other services
with open("config.yaml") as file:
    config = yaml.safe_load(file)

# Retrieve the Slack-related parameters
slack_params = config["slack"]
slack_bot_token = slack_params["slack_bot_token"]
slack_app_token = slack_params["slack_app_token"]

# Retrieve other parameters
openai_key = config["openai_model_api"]["api_key"]
log_path = config["files"]["log_path"]

# Set up environmental variables for Slack API and OpenAI access
os.environ["SLACK_BOT_TOKEN"] = slack_bot_token
os.environ["SLACK_APP_TOKEN"] = slack_app_token
os.environ["OPENAI_API_KEY"] = openai_key


# Create logger, slack app, slack client, and Virtual Havruta instance
vh_logger = create_logger(f=log_path)
app = App(token=os.environ.get("SLACK_BOT_TOKEN"), logger=vh_logger)
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
vh = VirtualHavruta('prompts.yaml', 'config.yaml', vh_logger)

# Slack Pipeline
def slack_post_msg(slack_client, channel_id, message_ts, msg, logger, kg_msg=''):
    attachments = []
    if kg_msg:
        attachments = [
                {
                    "fallback": "",
                    "blocks": [
                        {
                            "type": "section",
                            "block_id": "section2",
                            "text": {"type": "mrkdwn", "text": kg_msg},
                            "accessory": {
                                "type": "image",
                                "image_url": "https://miro.medium.com/v2/resize:fit:1400/0*D4ZRtctQTvoNedQY",
                                "alt_text": "KG",
                            },
                        },
                    ],
                }
            ]
    
    result = slack_client.chat_postMessage(
        channel=channel_id,
        thread_ts=message_ts,
        text=msg,
        attachments=attachments,
        as_user=True,
    )
    logger.info(
            f"[SLACK POST MESSAGE] ChannelID= {channel_id}. MsgID={channel_id}-{message_ts}. Msg={msg}. KGLINK={kg_msg}"
            f"RESULT={result}."
        )

@app.event("message")
def slack_pipeline(body, logger):
    # Log the raw message body
    logger.info(body)

    # Extract relevant information from the incoming Slack message
    query = body["event"]["text"]
    channel_id = body["event"]["channel"]
    message_ts = body["event"]["ts"]
    
    # Create a unique message identifier using channel ID and timestamp
    msgid = f"{channel_id}-{message_ts}"
    # Clean the query from special Slack characters
    query = part_res(query, '>')
    logger.info(f"MsgID={msgid} [RECEIVED QUERY] QUERY={query}")

    global vh  # Global variable for the virtual havruta instance

    # Check if the message is a reply to another message or a new mention
    is_reply = "thread_ts" in body["event"]

    # Skip processing if the query contains Slack notifications or is a reply
    if (
        "joined #virtual-havruta" not in query
        and "joined #general" not in query
        and "<!channel" not in query
        and "<!here" not in query
        and "<!everyone" not in query
        and not is_reply
    ):
        try:
            total_tokens = 0  # Initialize token counter

            # Post an initial message acknowledging the received question
            interim_msg = "I got your question. Let me get started to think."
            slack_post_msg(client, channel_id, message_ts, interim_msg, logger)
            
            '''
            ####################
            Insert customized response functionalities here
            ####################
            '''

            # Log the total number of tokens spent processing this query
            logger.info(f"MsgID={msgid}. [Token Count] Spent {total_tokens} tokens in total for this round of query.")
        except Exception as e:
            logger.error(f"MsgID={msgid}. [Pipeline Error] {e}")
            slack_post_msg(client, channel_id, message_ts, 'An error occurred while I tried to answer. Please contact administrator.', logger)


# Start the Slack app in socket mode to listen for events
if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

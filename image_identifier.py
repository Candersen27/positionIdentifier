from openai import OpenAI

from dotenv import load_dotenv
import base64, os, requests
import ast

# Functions

# helper function that encodes an image for LLM consumption
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def process_image_with_llm(image_path):
    
    # LLM prompt
    system_prompt = """f
    You are a computer vision model that has been trained to identify baseball players, umpires, and coaches.
    You are given an image of a person, and you are asked to identify the position of the player using the following guidelines:
    ###
    1: "Pitcher",
    2: "Batter",
    3: "Catcher",
    4: "Umpire",
    5: "Coach",
    6: "Fielder",
    7: "Unknown"
    ###
    You will give your response in the form of a list of two numbers.  
    The first number is a single digit from 1 to 7, where 1 is the pitcher, 2 is the batter, 3 is the catcher, 4 is the umpire, 5 is the coach, 6 is the fielder, and 7 is unknown.
    The second number is a confidence score from 0 to 1, rounded to two significant figures. The more confident you are in your answer, the higher this value should be.
    Only output the two numbers in a list format, nothing else.
    """

    encoded_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    } 

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 9,
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers= headers, json=payload)
    choices = response.json().get('choices', [])


    return choices[0].get('message', {}).get('content', '')






def get_position(image_path):
    response = process_image_with_llm(image_path)
    response = ast.literal_eval(response)
    return response


def main():
    # load the env file and get the openai api key
    load_dotenv()
    #LLM_client = OpenAI(os.getenv('OPENAI_API_KEY'))
    images_dir = os.path.join(os.path.dirname(__file__), 'images')
    
    
    catcher_image_path = os.path.join(images_dir, 'catcher.PNG')
    pitcher_image_path = os.path.join(images_dir, 'pitcher.jpg')
    batter_image_path = os.path.join(images_dir, 'hitter.PNG')
    umpire_image_path = os.path.join(images_dir, 'umpire.PNG')
    coach_image_path = os.path.join(images_dir, 'coach.PNG')
    fielder_image_path = os.path.join(images_dir, 'ss.PNG')
    mascot_image_path = os.path.join(images_dir, 'mascot.PNG')

    print(f"Catcher: {get_position(catcher_image_path)}")
    print(f"Pitcher: {get_position(pitcher_image_path)}")
    print(f"Batter: {get_position(batter_image_path)}")
    print(f"Umpire: {get_position(umpire_image_path)}")
    print(f"Coach: {get_position(coach_image_path)}")
    print(f"Fielder: {get_position(fielder_image_path)}")
    print(f"Mascot: {get_position(mascot_image_path)}")



if __name__ == "__main__":
    main()


import requests


def api_get_loss(input_str):
    """
    Returns a list of losses for each token
    :param input_str: string
    :return: List[float]
    """
    output = api_call(input_str)
    try:
        filtered_out = output[0]["details"]["prefill"][1:]
        loss = [-1 * i["logprob"] for i in filtered_out]
    except:
        import pdb
        pdb.set_trace()
    return loss

def api_call(inputs):
    """
    Calls the Huggingface API for the given input
    :param input: string
    :return: api return object
    """

    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": inputs,
        "parameters": {"details": True, "max_new_tokens": 1}
    })

    return output
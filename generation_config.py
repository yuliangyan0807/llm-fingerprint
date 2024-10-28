from generation import *
from test_prompt import *
import logging
from transformers import set_seed

# set_seed(42)

logging.basicConfig(level=logging.INFO,
                    filename='generation_config.log',
                    filemode='a',
                    format='%(message)s'
                    )

if __name__ == '__main__':
    prompt = EXAMPLE_3
    for i in range(10):
        output = generation("/home/yuliangyan/Code/llm-fingerprinting/llama3_1_8_ft_tiny_test",
                            temperature=0.000001,
                            prompt=prompt)
        logging.info(output)
        logging.info("###################################################")
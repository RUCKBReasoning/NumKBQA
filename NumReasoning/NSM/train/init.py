import numpy as np
from NSM.Agent.NSMAgent import NsmAgent
from functools import reduce


def init_nsm(args, logger, num_entity, num_relation, num_word):
    logger.info("Building {}.".format("Agent"))
    agent = NsmAgent(args, logger, num_entity, num_relation, num_word)
    logger.info("Architecture: {}".format(agent))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in agent.parameters()])
    logger.info("Agent params: {}".format(total_params))

    return agent

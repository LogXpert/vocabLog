# SPDX-License-Identifier: MIT
# This file implements the TOOLS algorithm for log parsing.

from abc import ABC, abstractmethod
import token
from typing import cast, Collection, IO, Iterable, List, MutableMapping, MutableSequence, Optional, Sequence, Tuple, \
    TYPE_CHECKING, TypeVar, Union
import random
import string

from simpleProfiler import Profiler, NullProfiler
from collections import defaultdict
import logging
import logging.config
import re
import nltk
import en_core_web_md
#import en_core_web_lg
import math
import json
import requests
import os
from openai import OpenAI
import tiktoken
import torch
import math
from collections import Counter


#nlp = en_core_web_lg.load()
nlp = en_core_web_md.load()

if os.path.exists("./pos.log"):
    os.rename("./pos.log", "./pos.log.prev")
pos_logger = logging.getLogger("pos_logger")
pos_handler = logging.FileHandler("./pos.log", mode="w")
pos_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

# Define pairs of symmetric symbols
SYMMETRIC_PAIRS = {
    #'{': '}',
    '[': ']',
    '(': ')',
    #'<': '>'
}
OPENING = set(SYMMETRIC_PAIRS.keys())
CLOSING = set(SYMMETRIC_PAIRS.values())

class TemplateInfo():
    def __init__(self, templateId: int, templateStr: str) -> None:
        self.templateId = templateId
        self.templateStr = templateStr
        self.logIndexes = []
        self.matchedLogSize = 0

class VocabLog():
    def __init__(self,
                 depth: int = 4,
                 sim_th: float = 0.4,
                 max_children: int = 100,
                 max_clusters: Optional[int] = None,
                 extra_delimiters: Sequence[str] = (),
                 profiler: Profiler = NullProfiler(),
                 param_str: str = "<*>",
                 parametrize_numeric_tokens: bool = True,
                 LLM_support: bool = False,
                 LLM_provider: str = "openai",
                 LLM_model: str = "gpt-3.5-turbo",
                 LLM_api_key: Optional[str] = None,
                 LLM_thinking: bool = False,
                 similarityMeasure: str = "lcs"
                 ) -> None:
        """
        Create a new vocabLog instance.

        :param depth: max depth levels of log clusters. Minimum is 3.
            For example, for depth==4, Root is considered depth level 1.
            Token count is considered depth level 2.
            First log token is considered depth level 3.
            Log clusters below first token node are considered depth level 4.
        :param sim_th: similarity threshold - if percentage of similar tokens for a log message is below this
            number, a new log cluster will be created.
        :param max_children: max number of children of an internal node
        :param max_clusters: max number of tracked clusters (unlimited by default).
            When this number is reached, model starts replacing old clusters
            with a new ones according to the LRU policy.
        :param extra_delimiters: delimiters to apply when splitting log message into words (in addition to whitespace).
        :param parametrize_numeric_tokens: whether to treat tokens that contains at least one digit
            as template parameters.
        """
        if depth < 3:
            raise ValueError("depth argument must be at least 3")

        #self.log_cluster_depth = depth
        #self.max_node_depth = depth - 2  # max depth of a prefix tree node, starting from zero
        self.sim_th = sim_th
        #self.max_children = max_children
        #self.root_node = Node(NodeType.ROOT)
        self.profiler = profiler
        self.extra_delimiters = extra_delimiters
        self.max_clusters = max_clusters
        self.param_str = param_str
        self.parametrize_numeric_tokens = parametrize_numeric_tokens
        self.LLM_support = LLM_support
        self.LLM_provider = LLM_provider
        self.LLM_model = LLM_model
        self.LLM_api_key = LLM_api_key
        self.LLM_thinking = LLM_thinking
        self.templateId = 0
        self.llmCallCnt = 0
        self.llmCallTokens = 0

        #self.posSeqToTemplateInfos: MutableMapping[Sequence[str], List[TemplateInfo]] = {}
        self.posSeqToTemplateInfos: MutableMapping[str, List[TemplateInfo]] = defaultdict(list)
        self.idToTemplateStr: MutableMapping[int, str] = {}
        self.idToTemplateInfos: MutableMapping[int, TemplateInfo] = {}
        self.similarityMeasure = similarityMeasure

    def getNewTemplateId(self) -> int:
        self.templateId += 1
        return self.templateId

    def split_string(self, s):
        # First, replace any hyphens with spaces
        s = s.replace("-", "__DASH__")
        punctuation_pattern = r'^[,.\!?;:]+|[,.\!?;:]+$'
        match = re.match(punctuation_pattern, s)
        if match:
            front_punctuation = match.group(0)
        else:
            front_punctuation = ''

        stripped_s = re.sub(r'^[,.\!?;:]+', '', s)
        match = re.search(punctuation_pattern, stripped_s)
        if match:
            back_punctuation = match.group(0)
            stripped_s = re.sub(r'[,.\!?;:]+$', '', stripped_s)
        else:
            back_punctuation = ''

        result = []
        if front_punctuation:
            for char in front_punctuation:
                result.append(char)

        if '=' in stripped_s and ':' in stripped_s:
            if stripped_s.index('=') < stripped_s.index(':'):
                left, sep1, remainder = re.split(r"(=)", stripped_s, 1)
                mid, sep2, right = re.split(r"(:)", remainder, 1)
            else:
                left, sep1, remainder = re.split(r"(:)", stripped_s, 1)
                mid, sep2, right = re.split(r"(=)", remainder, 1)
            result.extend([left, sep1, mid, sep2, right])

        elif stripped_s.count('=') == 1 or (stripped_s.count('=') > 1 and not "==" in stripped_s):
            left, sep, right = re.split(r"(=)", stripped_s, 1)
            result.extend([left, sep, right])

        elif stripped_s.count(':') == 1 or (stripped_s.count(':') > 1 and not "::" in stripped_s):
            left, sep, right = re.split(r"(:)", stripped_s, 1)
            result.extend([left, sep, right])

        else:
            result.append(stripped_s)

        if back_punctuation:
            for char in back_punctuation:
                result.append(char)

        tokens = []
        for part in result:
            if (
                len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and part[-1] == SYMMETRIC_PAIRS[part[0]]):
                tokens.append(part[0])
                tokens.append(part[1:-1])
                tokens.append(part[-1])
            elif (len(part) > 1
                and part[0] in SYMMETRIC_PAIRS and not SYMMETRIC_PAIRS[part[0]] in part):
                tokens.append(part[0])
                tokens.append(part[1:])
            elif (len(part) > 1
                and part[-1] in CLOSING and \
                not any(key for key in SYMMETRIC_PAIRS if SYMMETRIC_PAIRS[key] == part[-1] and key in part[0:-1])):
                tokens.append(part[:-1])
                tokens.append(part[-1])
            elif len(part) > 2:
                prefixIdx = suffixIdx = 0
                splitDone = False
                i = 1
                while i < len(part):
                    if part[i] in SYMMETRIC_PAIRS:
                        prefixIdx = i
                        close_sym = SYMMETRIC_PAIRS[part[i]]
                        for k in range(prefixIdx+1, len(part)):
                            if part[k] == close_sym:
                                suffixIdx = k
                                tokens.append(part[0:prefixIdx])
                                tokens.append(part[prefixIdx])
                                tokens.append(part[prefixIdx+1:suffixIdx])
                                tokens.append(part[suffixIdx])
                                tokens.append(part[suffixIdx+1:])
                                splitDone = True
                                break
                        if splitDone == False:
                            tokens.append(part[0:prefixIdx+1])
                            tokens.append(part[prefixIdx+1:])
                            splitDone = True
                    elif part[i] in CLOSING:
                        tokens.append(part[0:i])
                        tokens.append(part[i])
                        tokens.append(part[i+1:])
                        splitDone = True

                    if splitDone == True:
                        break
                    else:
                        i += 1
                if not splitDone:
                    tokens.append(part)
            else:
                tokens.append(part)
        return tokens

    def isSnakeMode(self, token: str) -> bool:
        """
        Check if the token is in snake_case format.
        """
        return bool(re.match(r'^[a-zA-Z]+(_[a-zA-Z]+)+$', token))

    def isCamelMode(self, token: str) -> bool:
        """
        Check if the token is in camelCase format.
        """
        if not re.match(r'^[a-zA-Z]+([A-Z][a-z0-9]+)+(\(\))?$', token):
            return False
        parts = re.findall(r'[a-z]+|[A-Z][a-z0-9]*', token)
        if len(parts) < 2:
            return False
        for p in parts:
            if len(p) > 1 and not any(ch.isdigit() for ch in p) and nlp.vocab.has_vector(p):
                return True
        return False

    def is_snake_or_camel(self, identifier):
        snake_case_pattern = r'^[a-z]+(_[a-z]+)+$'
        camel_case_pattern = r'^[a-zA-Z]+([A-Z][a-z0-9]+)+$'
        return bool(re.match(snake_case_pattern, identifier) or re.match(camel_case_pattern, identifier))

    def isAllAlphaCapital(self, token: str) -> bool:
        if token == "<*>":
            return False

        if any(ch.isdigit() for ch in token):
            return False

        return all(char.isupper() for char in token if char.isalpha())

    def parseChunkLogs(self, logBase: int, chunkSize: int, chunkLogs: list):
        """
        Parse a chunk of logs and return the template IDs and their string representations.
        :param chunkLogs: List of log messages in the chunk.
        :return: Tuple containing a list of template IDs and a list of template strings.
        """
        logToParseClusters = defaultdict(list)
        for logIndex, logContent in enumerate(chunkLogs):
            logIndex = logBase * chunkSize + logIndex  # Adjust log index based on the base
            logTokens, tokenPosTag = self.extractTokensOfMsg(logContent)
            logContent = " ".join(logTokens)  # Concatenate tokens to form the new log
            staticTokens = []
            for i, token in enumerate(logTokens):
                if nlp.vocab.has_vector(token.strip()) and (i == 0 or (i > 0 and not logTokens[i-1] in {"=", ":", "for", "user", "to"})) and token not in {"true", "false"}:
                    staticTokens.append((i, token))
                elif self.isAllAlphaCapital(token):
                    if (i > 0 and i < len(logTokens)-1 and logTokens[i-1] in {"=", ":"} and logTokens[i+1] in {","}) or (i == len(logTokens)-1 and logTokens[i-1] in {"="}):
                        continue
                    staticTokens.append((i, token))
                elif (i == 0 and self.isSnakeMode(token)) or self.isCamelMode(token) or re.match(r"^[^:]+::[^:]+$", token):
                    staticTokens.append((i, token))
                elif i > 0 and i < len(logTokens)-1 and (nlp.vocab.has_vector(token)) and (logTokens[i+1] in {"=", ":"}):
                    staticTokens.append((i, token))

            if not staticTokens:
                staticTokensSequence = "<*>"
            else:
                staticTokensSequence = "".join(f"{idx}{token}" for idx, token in staticTokens)

            # check if log meesage match to specific template of list
            templateId = self.findMatchedTemplateFromCache(logContent)
            if templateId is not None:
                self.idToTemplateInfos[templateId].matchedLogSize += 1
                self.idToTemplateInfos[templateId].logIndexes.append(logIndex)
                continue
            else:
                # Store logIndex and logContent into logToParseClusters
                if staticTokensSequence != "<*>":
                    # If staticTokensSequence is not empty, create or use existing list with that key
                    logToParseClusters[staticTokensSequence].append((logIndex, logContent))
                else:
                    # If staticTokensSequence is empty or just "<*>", use "<*>" as key
                    logToParseClusters["<*>"].append((logIndex, logContent))

        # parsing start when all chunk logs are added to logToParseClusters
        # Iterate over each staticTokensSequence and process logs
        for staticTokensSequence, logIndexContents in logToParseClusters.items():
            if staticTokensSequence == '<*>':
                # generate logs similar matrix to better handle Variable-length log message
                logSimilarityMatrix = self.generateLogSimilarityMatrix(logIndexContents)
                # generate clusters based on logSimilarityMatrix
                logClusters = self.generateClustersFromSimilarityMatrix(logSimilarityMatrix, logIndexContents)
            else:
                logClusters = [list(range(len(logIndexContents)))]
            #generate template for each cluster
            for idx, cluster in enumerate(logClusters):
                # fetch log contents from cluster
                logContentsOfsameCluster = [logIndexContents[i][1] for i in cluster]  # Get log strings from the cluster, item of cluster is the index of logIndexContents:[(rawLogIndex, rawLogContent),...]
                logIndexesOfsameCluster = [logIndexContents[i][0] for i in cluster]  # Get log indexes from the cluster, item of cluster is the index of logIndexContents:[(rawLogIndex, rawLogContent),...]
                for logContent in logContentsOfsameCluster:
                    pass
                templatesList = self.extractTemplatesFromCluster(logContentsOfsameCluster, logIndexesOfsameCluster)
                for templateStr, logIndexes in templatesList:
                    #check whether templateStr can be merged with existing templates
                    #collect all templateStr from idToTemplateInfos
                    toMergeTemplateInfos = self.collectCanMergeTemplateInfos(templateStr)
                    if toMergeTemplateInfos:
                        candidateLogs = [templateInfo.templateStr for templateInfo in toMergeTemplateInfos]
                        candidateLogs.append(templateStr)  # Add the new template string to the candidate logs
                        newTemplateStr = self.extractTemplateByLLM(candidateLogs, 2)

                        # collect all logIndexes from toMergeTemplateInfos
                        mergedLogIndexes = []
                        for templateInfo in toMergeTemplateInfos:
                            mergedLogIndexes.extend(templateInfo.logIndexes)
                        mergedLogIndexes.extend(logIndexes)  # Add the new log indexes to the merged list
                        # create a new TemplateInfo with merged templateStr and logIndexes
                        templateInfo = self.createNewTemplateInfo(newTemplateStr, mergedLogIndexes)
                        self.idToTemplateInfos[templateInfo.templateId] = templateInfo
                        # remove toMergeTemplateInfos from idToTemplateInfos
                        for templateInfo in toMergeTemplateInfos:
                            del self.idToTemplateInfos[templateInfo.templateId]
                    else:
                        templateInfo = self.createNewTemplateInfo(templateStr, logIndexes)
                        self.idToTemplateInfos[templateInfo.templateId] = templateInfo
        #for i, info in self.idToTemplateInfos.items():

    def getAllLogTemplates(self):
        logTemplateSummary = {}
        for id, info in self.idToTemplateInfos.items():
            for logIndex in info.logIndexes:
                #pos_logger.debug(f"Assigning template ID: {id} to raw log index: {logIndex}")
                logTemplateSummary[logIndex] = (id, info.templateStr)
        # Sort by log index and extract ids and template strings to separate lists
        sorted_log_indices = sorted(logTemplateSummary.keys())
        template_ids = []
        template_strs = []

        for idx in sorted_log_indices:
            template_id, template_str = logTemplateSummary[idx]
            template_ids.append(template_id)
            template_strs.append(template_str)

        return template_ids, template_strs

    def collectCanMergeTemplateInfos(self, templateStr: str) -> list[TemplateInfo]:
        """
        Collect all TemplateInfo objects that can be merged with the given templateStr.
        :param templateStr: The template string to check for merging.
        :return: List of TemplateInfo objects that can be merged with the given templateStr.
        """
        canMergeTemplateInfos = []
        for templateInfo in self.idToTemplateInfos.values():
            # Check if the new template can be merged with the existing one
            oldTokens = templateInfo.templateStr.split()
            newTokens = templateStr.split()
            if abs(len(oldTokens) - len(newTokens)) > 2:
                continue
            elif len(oldTokens) == len(newTokens):
                # Check if the tokens match or can be merged
                for oldToken, newToken in zip(oldTokens, newTokens):
                    if oldToken != newToken and oldToken != "<*>" and newToken != "<*>":
                        break
                else:
                    # If we didn't break, it means all tokens match or can be merged
                    canMergeTemplateInfos.append(templateInfo)

            elif self.isTemplateMached(templateInfo.templateStr, templateStr) or self.isTemplateMached(templateStr, templateInfo.templateStr):
                canMergeTemplateInfos.append(templateInfo)

        return canMergeTemplateInfos
    def mergeTemplates(self, existingTemplateStr: str, newTemplateStr: str) -> Tuple[bool, str]:
        """
        Merge two templates if they can be combined.
        :param existingTemplateStr: The existing template string.
        :param newTemplateStr: The new template string to merge.
        :return: Tuple indicating if merge happened and the merged template string.
        """
        # Check if the new template can be merged with the existing one
        oldTokens = existingTemplateStr.split()
        newTokens = newTemplateStr.split()
        if len(oldTokens) != len(newTokens):
            return False, newTemplateStr
        # Check if the tokens match or can be merged
        mergedTokens = []
        for oldToken, newToken in zip(oldTokens, newTokens):
            if oldToken == newToken:
                mergedTokens.append(oldToken)
            elif oldToken == "<*>" or newToken == "<*>":
                mergedTokens.append("<*>")  # Use wildcard if either token is a wildcard
            else:
                return False, newTemplateStr  # If tokens differ and are not wildcards, cannot merge
        return True, " ".join(mergedTokens)

    def extractTemplatesFromCluster(self, logContents: list[str], logIndexes: list[int]):
        """
        Extract templates from a cluster of logs.
        :param cluster: List of tuples containing log index and log content.
        :return: List of (TemplateStr, [log index]) objects created from the cluster.
        """
        candidateLogs = []
        templatesList = []
        # split each log string with space into tokens
        logTokens = [logString.split() for logString in logContents]
        # count every logToken length and give out the number of logs with specific length
        logTokenLengths = [len(tokens) for tokens in logTokens]
        logTypeNum = len(set(logContents))
        if logTypeNum == 1:
            templateStr = logContents[0]
            return [(templateStr, logIndexes)]  # Return the template string and the log indexes
        # Check if all logs have the same length
        if len(set(logTokenLengths)) != 1:
            for length in set(logTokenLengths):
                logIds = [i for i, l in enumerate(logTokenLengths) if l == length]
                candidateLogs.append(logContents[logIds[0]])  # Use the first log with this length as a candidate
            templateStr = self.extractTemplateByLLM(candidateLogs, 2)
            if templateStr is None:
                templatesList = self.getTemplatesByUniqueLog(logContents, logIndexes)
            else:
                templatesList = [(templateStr, logIndexes)]
            return templatesList
        else:  # If all logs have the same length, use frequency / entropy to judge the template
            #templatesList = self.getTemplatesByFrequency(logTypeNum, logContents, logIndexes)
            if not templatesList:
                # transpose logTokens to get tokens per position, if there is <*> in positionTokensList, all tokens in this position are changed to wildcard
                positionTokensList = [list(col) for col in zip(*logTokens)] # Transpose to get tokens per position
                for i, positionTokens in enumerate(positionTokensList):
                    if "<*>" in positionTokens:
                        positionTokensList[i] = ["<*>"] * len(positionTokens)
                #calculate entropy for each position with positionTokensList
                entropyList = []
                for i, positionTokens in enumerate(positionTokensList):
                    entropy = self.shannon_entropy(positionTokens)
                    entropyList.append(entropy)
                #loop each position and check if entropy is low enough to use the token as template token
                templateToken = []
                LLMisNeeded = False  # Flag to indicate if LLM is needed for template
                OneTemplateExtracted = True
                for pos, ent in enumerate(entropyList):
                    if ent == 0:
                        templateToken.append(positionTokensList[pos][0])
                    elif any(not nlp.vocab.has_vector(token) for token in positionTokensList[pos]) and not pos == 0 and ent > 1.8:
                        templateToken.append(self.param_str)
                    elif pos > 0 and nlp.vocab.has_vector(logTokens[0][pos]) and logTokens[0][pos-1] in {"="}:
                        templateToken.append(self.param_str)
                    else:
                        LLMisNeeded = True
                        OneTemplateExtracted = False
                        #break #contine change all log's positionTokens to wildcard if one log has it.
                if OneTemplateExtracted:
                    templateStr = " ".join(templateToken)
                    return [(templateStr, logIndexes)]  # Return the template string and the log indexes
                if LLMisNeeded:
                    # at most random sample 5 logs from logContents to extract template, use all if less than 5
                    modifiedLogTokens = [list(row) for row in zip(*positionTokensList)]
                    logContents = [" ".join(tokens) for tokens in modifiedLogTokens]
                    logCounts = Counter(logContents) # logCount: [(log, count), ...]
                    num = 0
                    sampleLogStrings = []
                    for log, count in logCounts.items():
                        sampleLogStrings.append(log)
                        num += 1
                        if num >= 5:  # Limit to 5 samples
                            break
                    #sampleLogStrings = random.sample(logContents, min(5, len(logContents)))
                    #pos_logger.debug(f"Using {len(sampleLogStrings)} logs to extract template by LLM.")
                    templateStr = self.extractTemplateByLLM(sampleLogStrings)
                    if templateStr == "":
                        templatesList = self.getTemplatesByUniqueLog(logContents, logIndexes)
                    else:
                        templatesList = [(templateStr, logIndexes)]
                    return templatesList

    def getTemplatesByUniqueLog(self, logContents: list[str], logIndexes: list[int]) -> list[Tuple[str, list[int]]]:
        """ Get the unique logs and their indexes as templates."""
        uniqueLogs = set(logContents)
        templateGroups = defaultdict(list)
        for idx, log in enumerate(logContents):
            templateGroups[log].append(logIndexes[idx])  # Group log indexes by log content
        templatesList = []
        for templateStr, logIndexes in templateGroups.items():
            templatesList.append((templateStr, logIndexes))  # Append the template string and its corresponding log indexes
        return templatesList

    def getTemplatesByFrequency(self, logTypeNum, logContents: list[str], logIndexes: list[int]) -> list[Tuple[str, list[int]]]:
        """ Get the maximum frequency of each log in the log contents."""
        logCounts = Counter(logContents) # logCount: [(log, count), ...]
        total = len(logContents)
        maxFreq = 0
        maxFreqLog = None
        for log, count in logCounts.items():
            proportion = count / total
            if proportion > maxFreq:
                maxFreq = proportion
                maxFreqLog = log
            print(f"Log: {log}\nCount: {count}, Proportion: {proportion:.2%}\n")
        if maxFreq >= 1/logTypeNum:
            templateGroups = defaultdict(list)
            for idx, log in enumerate(logContents):
                templateGroups[log].append(logIndexes[idx])  # Group log indexes by log content
            templatesList = []
            for templateStr, logIndexes in templateGroups.items():
                templatesList.append((templateStr, logIndexes))  # Append the template string and its corresponding log indexes
            return templatesList
        return []

    def createNewTemplateInfo(self, templateStr: str, logIndexes: list[int]) -> Optional[TemplateInfo]:
        templateInfo = TemplateInfo(self.getNewTemplateId(), templateStr)
        templateInfo.matchedLogSize += len(logIndexes)
        templateInfo.logIndexes.extend(logIndexes)
        return templateInfo

    def extractTemplateFromCluster(self, logStrings: list) -> str:
        """
        Extract a template from a list of log contents.
        :param logContents: List of log contents in the cluster.
        :return: Template object created from the log contents.
        """
        candidateLogs = []
        # split each log string with space into tokens
        logTokens = [logString.split() for logString in logStrings]
        # count every logToken length and give out the number of logs with specific length
        logTokenLengths = [len(tokens) for tokens in logTokens]
        if len(set(logStrings)) == 1:
            templateStr = logStrings[0]
            return templateStr
        # Check if all logs have the same length
        if len(set(logTokenLengths)) != 1:
            for length in set(logTokenLengths):
                logIds = [i for i, l in enumerate(logTokenLengths) if l == length]
                candidateLogs.append(logStrings[logIds[0]])  # Use the first log with this length as a candidate
            templateStr = self.extractTemplateByLLM(candidateLogs)
        else:            # If all logs have the same length, use entropy to judge the template
            entropyPerPosition, positionTokensList = self.compute_position_entropy(logTokens)
            templateToken = []
            LLMisNeeded = False  # Flag to indicate if LLM is needed for template
            for pos, ent in enumerate(entropyPerPosition):
                if '<*>' in positionTokensList[pos]:  # If the token is a wildcard, use it as the template token
                    templateToken.append('<*>')
                elif nlp.vocab.has_vector(logTokens[0][pos]) and ent <= 1:
                    templateToken.append(logTokens[0][pos])  # Use the token from the first log as the template token
                elif not nlp.vocab.has_vector(logTokens[0][pos]) and ent >= 3:
                    templateToken.append(self.param_str)
                else: # cannot judge, use LLM
                    LLMisNeeded = True
            if LLMisNeeded:
                # at most random sample 5 logs from logStrings to extract template, use all if less than 5
                sampleLogStrings = random.sample(logStrings, min(5, len(logStrings)))
                templateStr = self.extractTemplateByLLM(sampleLogStrings)
            else:
                templateStr = " ".join(templateToken)
        return templateStr

    def shannon_entropy(self, tokens):
        total = len(tokens)
        counts = Counter(tokens)
        return -sum((count/total) * math.log2(count/total) for count in counts.values())

    def compute_position_entropy(self, logs):
        transposed = list(zip(*logs))
        entropyList = []
        positionTokensList = []
        for i, position_tokens in enumerate(transposed):
            entropy = self.shannon_entropy(position_tokens)
            entropyList.append(entropy)
            positionTokensList.append(position_tokens)
        return entropyList, positionTokensList

    def extractTemplateByLLM(self, logs: list, mode: int = 1) -> str:
        """
        Extract a template from the candidate logs using a language model.
        :param candidateLogs: List of candidate log contents.
        :return: Template object created from the candidate logs.
        """
        # prompt format: instruction + (demonstration) + query(logs)
        instruction = "You are a log parse assistant to help extract log template from input logs."
        inputLogMsgs = '\n' + '\n'.join([f'`{log}`'for log in logs])

        if mode == 1:
            prompt = f"""
            Your task is to abstract log template from input log messages, variable part is marked with `<*>`.
            You should decide whether the different token in same position is a constant or variable with following rules:

            ## Rules for keep tokens as constants:
            - If a token is an **adjective, adverb, verb, or proper noun or domain-specific term(e.g., `HTTPS`, `IPv4`, `BSSID`, `SCREEN_ON`, `SCREEN_OFF`, `JOB_SETUP`)**, keep it as a constant.
            - If a token is a **modifier in a compound noun**(e.g., `Removable`, `illegal`, `Idle`, `Active`) and the modifier **represents a fixed attribute or label**, keep the compound noun as a constant.
            - If a token serves as the subject in a sentence with a subject-verb-object structure to represent a configuration key or property name(e.g., `cpusd`, `Failed none`), then it must be treated as **constants** and not abstracted.
            - If a token **reflects behavioral information(e.g., `Accepted`, `Failed`, `Timeout`)**  — keep it as a constant.
            - If a token **reflects distinctly different or opposing behaviors** (e.g., `boot` vs `shutdown`) — keep it as a constant.
            - Do not treat "user" and "users", "service" and "services", or other singular/plural variants as the same — keep it as a constant.
            - If a token appears as the value in a key:value or key=value pair, abstract it as <*> while keeping the key constant — unless the value is domain-specific (e.g., SCREEN_ON) or a proper noun.
            - If the value is a boolean (e.g., true, false), always abstract it as <*>, keeping the key unchanged.
            - Replace variable parts (IDs, numbers, IPs, timestamps, user names, etc.) with wildcards `<*>`, e.g., `a1`, `B2` etc.

            ## Step 1: Token-by-Token Template Comparison

            Iterate over all token positions.
            - For each position where the tokens **differ**, generate a separate JSON object (Output-1) for this position, following the format below.
            - **Skip positions** where all logs share the same token.

            Output-1 Format:
            {{
                "position": <index of the token position>,
                "tokens": [list of tokens at this position across all logs],
                "Explanation": "Your reasoning for labeling as constant or variable, based on the rules.",
                "result": "constant" or "variable"
            }}

            ## Step 2: Template Extraction

            - If **any** position in Output-1 is labeled as `"constant"`, the logs do **not** share the same template. Return an **empty** template.
            - If **all** differing positions are labeled as `"variable"`, the logs share the same template. Replace variable tokens with `<*>` and return the template string.

            Output-2 Format:
            If logs share the same template:
            {{
            "template": "your abstracted template string, e.g., \"Removable disk <*> is <*>\""
            }}

            If logs do **not** share the same template:
            {{
            "template": ""
            }}

            ## Input Log Messages:
            {inputLogMsgs}

            Please generate ordered Output-1, Output-2 result seperately as described above with strictly valid JSON (not Python-style), parseable by json.loads:
            Output-1: <your JSON result for Step 1>
            Output-2: <your JSON result for Step 2>

            """
            #only output **Output-2**
        else:
            prompt = f"""
            Your task is to abstract log template from input log messages with different log length.
            The common part of logs are abstracted as template, the different part are abstracted as variable marked with `<*>`. use one <*> to represent consective duplicate part.
            Output Format:
            {{
            "template": "your abstracted template string, e.g., \"Removable disk <*> is <*>\""
            }}

            ## Input Log Messages:
            {inputLogMsgs}

            Please generate Output result as described above with strictly valid JSON (not Python-style), parseable by json.loads.
            """

        print(f"LLM input messages: {inputLogMsgs}")
        response = self.call_llm(prompt, instruction)
        print("==== LLM JSON response ===")
        print(f"LLM JSON response: {response}")
        print("===========================")

        try:
            # If response is a string containing JSON
            if isinstance(response, str):
                try:
                    # Try to parse it as JSON
                    parsed_response = json.loads(response)
                    templates = parsed_response.get("template", "")
                    for i, tpl in enumerate(templates, 1):
                        print(f"Template {i}: {tpl}")
                    return templates[0] if templates else ""
                except json.JSONDecodeError:
                    # If it's not valid JSON, try to extract JSON using regex
                    matches = re.findall(r'\{.*?\}', response, re.DOTALL)
                    if matches:
                            json_str = matches[-1]
                            parsed_response = json.loads(json_str)
                            template = parsed_response.get("template", "")
                    else:
                        template = ""
            # If response is already a dictionary
            elif isinstance(response, dict):
                template = response.get("template", "")
                #for i, tpl in enumerate(templates, 1):
                #    print(f"Template {i}: {tpl}")
                return template if template else ""
            else:
                template = ""
            return template
        except Exception as e:
            return ""

    def generateClustersFromSimilarityMatrix(self, logSimilarityMatrix: list, logs: list[Tuple[int, str]]) -> list:
        """
        Generate clusters from the similarity matrix based on the logs.
        :param logSimilarityMatrix: Similarity matrix as a list of lists.
        :param logs: List of tuples containing log index and log content.
        :return: List of clusters generated from the similarity matrix.
        """
        import networkx as nx
        import numpy as np

        # Build graph
        graph = nx.Graph()
        numLogs = len(logs)  # Use the number of logs as the number of nodes
        graph.add_nodes_from(range(numLogs))

        # Add edges based on similarity (1 means similar)
        for i in range(numLogs):
            for j in range(i+1, numLogs):
                if i < len(logSimilarityMatrix) and j < len(logSimilarityMatrix[i]) and logSimilarityMatrix[i][j] == 1:
                    graph.add_edge(i, j)

        # Get connected components (each component is a log cluster)
        logClusters = list(nx.connected_components(graph))

        # Output clustering results
        for clusterIdx, cluster in enumerate(logClusters): # cluster: [0, 1, 2,....] index of logs
            logIndices = [logs[i][0] for i in cluster]  # Get rawlog indices from the cluster

        return logClusters

    def generateLogSimilarityMatrix(self, logs: list) -> list:
        """
        Generate a similarity matrix for the given logs based on their static tokens.
        :param logs: List of tuples containing log index and log content.
        :return: Similarity matrix as a list of lists.
        """
        # Initialize an empty matrix
        matrix = [[0] * len(logs) for _ in range(len(logs))]

        # Compare each log with every other log
        for i, (logIndex1, logContent1) in enumerate(logs):
            for j, (logIndex2, logContent2) in enumerate(logs):
                #pos_logger.debug(f"Comparing matrix[{i}][{j}] / log {logIndex1} with log {logIndex2}")
                if i != j:
                    # Calculate similarity score between logContent1 and logContent2
                    similarityScore = self.calculateSimilarity(logContent1.split(), logContent2.split())
                    #pos_logger.debug(f"original similar score: {similarityScore}")
                    if self.sim_th == 0.0:
                        similarityScoreThres = 1 - (math.log(len(logContent1.split()), 2) / len(logContent1.split()))
                        similarityScore = 1 if similarityScore >= similarityScoreThres else 0
                        #pos_logger.debug(f"Dynamic similarity threshold: {similarityScoreThres}, corrected score is:{similarityScore}")
                    else:
                        similarityScore = 1 if similarityScore >= self.sim_th else 0
                else:
                    similarityScore = 1  # Similarity with itself is always 1
                # Fill the matrix with the similarity score
                #pos_logger.debug(f"Filling matrix[{i}][{j}] / logs[{logIndex1}][{logIndex2}] with score: {similarityScore}")
                matrix[i][j] = similarityScore
        return matrix

    def calculateSimilarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate similarity score between two token lists.
        :param tokens1: First list of tokens.
        :param tokens2: Second list of tokens.
        :return: Similarity score as a float.
        """
        if self.similarityMeasure == "jaccard":
            # Jaccard similarity
            set1 = set(tokens1)
            set2 = set(tokens2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union == 0:
                return 0.0
            return intersection / union
        if self.similarityMeasure == "cosine":
            return 1
        if self.similarityMeasure == "lcs":
            str1 = " ".join(tokens1)
            str2 = " ".join(tokens2)
            if str1 == str2:  # If both strings are identical, return 1.0
                return 1.0
            longerStr = str1 if str1 == max(str1, str2, key=len) else str2
            shorterStr = str2 if longerStr == str1 else str1
            if shorterStr in longerStr:  # If shorter string is a substring of longer string, return 1.0
                return 1.0
            if abs(len(tokens1) - len(tokens2)) > min(len(tokens1), len(tokens2)):  # If the difference in length is too large, return 0
                return 0.0
            lcsSeq = self.LCS(tokens1, tokens2)
            return len(lcsSeq) / min(len(tokens1), len(tokens2)) if min(len(tokens1), len(tokens2)) > 0 else 0.0

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1!=0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0,seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def findMatchedTemplateFromCache(self, logContent: str) -> int:
        """
        Find a matched template ID from the cache based on the static tokens sequence.
        :param logContent: The original log content.
        :return: Matched template ID or -1 if not found.
        """
        sortedTemplates = sorted(self.idToTemplateInfos.values(), key=lambda x: x.matchedLogSize, reverse=True)
        for templateInfo in sortedTemplates:
            # Check if the template matches the log content
            if self.isTemplateMached(logContent, templateInfo.templateStr):
                return templateInfo.templateId
        return None

    def isTemplateMached(self, logContent: str, templateStr: str) -> bool:
        """
        Check if the log content matches the template string.
        :param logContent: The original log content.
        :param templateStr: The template string to match against.
        :return: True if the log content matches the template, False otherwise.
        """
        log = re.sub(r'\s+', ' ', logContent.strip()) # DS
        pattern_parts = templateStr.split("<*>")
        pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
        regex_pattern = "(.*?)".join(pattern_parts_escaped)
        regex = "^" + regex_pattern + "$"
        matches = re.search(regex, log)
        if matches:
            wildcardValues = matches.groups()
            for value in wildcardValues:
                tokens = value.split()
                for token in tokens:
                    if nlp.vocab.has_vector(token) and (token.isalpha() or token in [',', '.', ';', ':']):#exclude {}()in string.punctuation):
                    #if not nlp.vocab.has_vector(token) and token != '<*>':
                        return False
                        #return True
                    #if token == '<*>':
                    #    pos_logger.debug(f"Token '{token}' is a wildcard. match")
                    #    return True
            return True
        else:
            return False
        if matches:
            return True #matches.groups()
        else:
            return False

    def extract_last_json(self, text):
        stack = []
        start_index = None
        last_json = None

        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start_index = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start_index is not None:
                        last_json = text[start_index:i+1]
        return last_json

    def parse_llm_stream(self, response):
        reasoning_log = []
        answer_log = []
        buffer = ""

        for chunk in response:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line.startswith("data:"):
                    continue

                json_str = line[len("data:"):].strip()
                if json_str == "[DONE]":
                    break

                try:
                    chunk_data = json.loads(json_str)
                except Exception as e:
                    print("JSON decode error:", e)
                    continue

                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                reasoning_chunk = delta.get("reasoning_content", "")
                answer_chunk = delta.get("content", "")

                if reasoning_chunk:
                    reasoning_log.append(reasoning_chunk)
                if answer_chunk:
                    answer_log.append(answer_chunk)

                if chunk_data.get("choices", [{}])[0].get("finish_reason") == "stop":
                    break

        return {
            "answer": ''.join(answer_log)
        }

    def call_llm(self, prompt: str, instruction: str = "") -> str:
        """Call an LLM API with the given prompt and return the response."""
        try:
            client = OpenAI(
                api_key=self.LLM_api_key,
                base_url=self.LLM_provider
            )

            if self.LLM_thinking:
                # Enable thinking mode if configured
                extra_body = {
                    "enable_thinking": True,
                    "thinking_budget": 4096,  # Set budget for thinking tokens
                    "stream": True  # Enable streaming for simplicity
                }
            else:
                # Disable thinking mode if not configured
                extra_body = {
                    "enable_thinking": False,
                    "stream": False  # Disable streaming for simplicity
                }
            if instruction == "":
                instruction = "You are a log parse assistant to help extract log template from input logs."

            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model = self.LLM_model,
                messages=messages,
                temperature = 0,
                extra_body=extra_body
            )
            if not response:
                return '{"template": ""}'

            log_content = ""
            if self.LLM_thinking:
                result = self.parse_llm_stream(response)
                log_content = result["answer"]
            else:
                # If not in thinking mode, just get the response content directly
                log_content = response.choices[0].message.content
            #print("LLM response contents: %s" % log_content)
            self.llmCallCnt += 1
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            prompt_token_ids = encoder.encode(prompt)
            response_token_ids = encoder.encode(log_content)
            self.llmCallTokens += (len(response_token_ids) + len(prompt_token_ids))
            # Try to parse JSON
            try:
                import re
                import json
                last_json_str = self.extract_last_json(log_content)
                if last_json_str:
                    parsed_response = json.loads(last_json_str)
                    template = parsed_response.get("template", "")
                    parsed = {"template": template}
                else:
                    return False, ""
            except (json.JSONDecodeError, TypeError) as e:
                print(f"\nWarning: Could not parse response as JSON: {e}")
                parsed = {"template": ""}
            return parsed

        except Exception as e:
            return '{"template": ""}'

    def extractTokensOfMsg(self, msg: str) -> Tuple[List[str], List[str]]:
        """
        Analyze text from input file using spaCy and NLTK.
        First preprocess special tokens, then analyze with spaCy.

        Args:
            msg (str): The message to analyze

        Returns:
            Tuple[List[str], List[str]]: Final tokens and their POS tags
        """
        #print(f"\nAnalyzing text: {msg}\n")

        # Step 1: Pre-process special tokens before spaCy tokenization
        # First split by whitespace to get initial tokens
        #initial_tokens = shlex.split(msg, posix=False)
        msg = msg.replace(",", ", ")
        initial_tokens = msg.split()

        # Process each token to handle special tokens and key-value patterns
        processed_tokens = []
        if len(initial_tokens) > 1:  # Only process if we have more than one token
            for token in initial_tokens:
                # Check if token contains any special token
                contains_special = False
                for special in ['<*>']:  # special_words:
                    if special in token:  # <*> is included into token
                        # token == <*>
                        if token == special:
                            processed_tokens.append(token)
                            contains_special = True
                            break

                        # token cover <*>, split token with pattern like 'key=<HEX>' or 'key:<HEX>'
                        elif '=' in token or ':' in token or '{' in token or '}' in token or '(' in token or ')' in token \
                            or '[' in token or ']' in token or ',' in token or '.' in token:
                            # Split on '=' or ':' first
                            parts = self.split_string(token)
                            # Replace any part that contains a special token
                            for i, part in enumerate(parts):
                            #for sp in special_words:
                                for sp in ['<*>']:
                                    if sp in part and part != sp:  # part has <*> inside, change part to <*>
                                        parts[i] = sp
                            processed_tokens.extend(parts)
                            contains_special = True
                            break
                        elif re.search(r'\.{2,}', token):
                            # Split the token by ellipsis pattern, but keep the ellipsis parts
                            parts = re.split(r'(\.{2,})', token)
                            # Filter out empty strings from the result
                            parts = [part for part in parts if part]
                            for i, part in enumerate(parts):
                                #for sp in special_words:
                                for sp in ['<*>']:
                                    if sp in part and part != sp:
                                        parts[i] = sp
                            processed_tokens.extend(parts)
                            contains_special = True
                            break
                        # If token contains special token but isn't exactly it
                        else:
                            processed_tokens.append(special)
                            #processed_tokens.append(token)
                            contains_special = True
                            break

                # If token doesn't contain any special token, process it normally
                if not contains_special:
                    parts = self.split_string(token)
                    processed_tokens.extend(parts)
        else:
            # If there's only one token or none, use them directly
            processed_tokens = initial_tokens

        result = []
        for token in processed_tokens:
            if token == "<*>" and result and result[-1] == "<*>":
                continue
            result.append(token)

        # Join processed tokens back into a string for spaCy analysis
        processed_text = " ".join(result)
        processed_text = processed_text.replace("__DASH__", "-")

        #======================
        result = processed_text.split()
        return result, []

# VocabLog
A Vocabulary-Driven and LLM-Augmented Framework for High-Performance Log Parsing

## Directory Structure

- `code/`: Core modules for masking, profiling, and template mining
- `evaluator/`: Evaluation scripts, configuration, and datasets

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/LogXpert/vocabLog.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. change evaluator/vocabLog.ini to set different configuration

4. Run dataset evaluation:
   ```bash
   python evaluator/evaluator.py


## Result

Result can be found in result.log under running directory

|    | Dataset     | Group Acc | PA Acc | FGA    | FTA    | Dur (sec) | #Line | #Lines/s | #Template | #LLM Call | #LLM Tokens |
|----|-------------|-----------|--------|--------|--------|-----------|-------|----------|-----------|-----------|-------------|
| 0  | HPC         | 0.992     | 0.9915 | 0.9318 | 0.7955 | 12.07     | 0     | 0        | 0         | 5         | 2395        |
| 1  | OpenStack   | 1         | 0.939  | 1      | 0.7907 | 9.56      | 0     | 0        | 0         | 2         | 1904        |
| 2  | BGL         | 0.9925    | 0.949  | 0.962  | 0.6582 | 23.15     | 0     | 0        | 0         | 6         | 4804        |
| 3  | HDFS        | 1         | 1      | 1      | 1      | 2.87      | 0     | 0        | 0         | 0         | 0           |
| 4  | Hadoop      | 0.9935    | 0.8965 | 0.9869 | 0.7773 | 7.74      | 0     | 0        | 0         | 1         | 887         |
| 5  | Spark       | 0.999     | 0.996  | 0.9577 | 0.845  | 6.27      | 0     | 0        | 0         | 1         | 969         |
| 6  | Zookeeper   | 0.9945    | 0.974  | 0.9608 | 0.902  | 9.3       | 0     | 0        | 0         | 3         | 1911        |
| 7  | Thunderbird | 0.9815    | 0.825  | 0.9044 | 0.6688 | 43.7      | 0     | 0        | 0         | 19        | 9155        |
| 8  | Windows     | 1         | 0.703  | 1      | 0.68   | 3.74      | 0     | 0        | 0         | 0         | 0           |
| 9  | Linux       | 0.998     | 0.9815 | 0.9741 | 0.7672 | 14.79     | 0     | 0        | 0         | 4         | 3614        |
| 10 | Andriod     | 0.9825    | 0.818  | 0.9587 | 0.7556 | 31.27     | 0     | 0        | 0         | 9         | 6261        |
| 11 | HealthApp   | 1         | 0.7495 | 1      | 0.9067 | 9.22      | 0     | 0        | 0         | 2         | 1807        |
| 12 | Apache      | 1         | 1      | 1      | 1      | 2.58      | 0     | 0        | 0         | 0         | 0           |
| 13 | Proxifier   | 1         | 1      | 1      | 1      | 2.87      | 0     | 0        | 0         | 0         | 0           |
| 14 | OpenSSH     | 1         | 0.9975 | 1      | 0.8462 | 20.12     | 0     | 0        | 0         | 10        | 3096        |
| 15 | Mac         | 0.968     | 0.65   | 0.9561 | 0.5526 | 50.91     | 0     | 0        | 0         | 19        | 12145       |
| 16 | Average     | 0.99      | 0.9    | 0.97   | 0.81   | 15.64     | 0     | 0        | 0         | 5.06      | 3059.25     |

## Usage

- Configure log parsing and template mining using `.ini` files in `evaluator/`.
- Use provided datasets for benchmarking and evaluation.
- Extend core modules in `code/` for custom log formats.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is open-source. See the LICENSE file for details.





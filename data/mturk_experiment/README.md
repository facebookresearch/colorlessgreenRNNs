Most of the files in this directory should not be distributed (we might process them in the future, if, for example, somebody is interested in the response time data). The only file to be publicly released is `clean_full_amazon_results.tab`. This file contains an extended version of the unaggregated Amazon Mechanical Turk results, with anonymized turker ids. The fields are:

- worker_id: unique subject ID (NOT the same as Amazon ID)
- type: target_generated, target_original or control
- answer_status: T or F
- answer: the target selected by the subject
- correct_form: the correct target
- opt1: the target presented first
- opt2: the target presented second
- full_id: sentence descriptive id
- sent: the sentence prefix


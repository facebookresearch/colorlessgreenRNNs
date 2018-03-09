The file `clean_full_amazon_results.tab` contains an extended version of the unaggregated Amazon Mechanical Turk results, with anonymized turker ids. (It is a superset of data used for LM evaluation.) The fields are:

- `worker_id`  unique subject ID (NOT the same as Amazon ID)
- `type`  *target_generated*, *target_original* or *control*
- `answer_status`  *T* or *F*
- `answer`  the target selected by the subject
- `correct_form`  the correct target
- `opt1`  the target presented first
- `opt2`  the target presented second
- `full_id`  sentence descriptive id
- `sent`  the sentence *prefix*


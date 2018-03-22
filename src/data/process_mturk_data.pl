#!/usr/bin/perl -w
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

while (<>) {

    chomp;
    # de-dosify just in case
    s/[\r\n]+//;
    # remove initial and final quotation marks
    s/^\"//;
    s/\"$//;
    # skip header
    if (/Description.*Keywords.*Reward/) {
	next;
    }
    # split remaining text into fields, delimited by ","
    @F = split /\",\"/,$_;
    # sanity check on number of fields
    if ($#F != 33) {
	print "the following line has $#F fields\n";
	print $_,"\n";
	exit;
    }
    # fields of interest:
    # 0	HITId
    # 1	HITTypeId
    # 2	Title
    # 3	Description
    # 4	Keywords
    # 5	Reward
    # 6	CreationTime
    # 7	MaxAssignments
    # 8	RequesterAnnotation
    # 9	AssignmentDurationInSeconds
    # 10	AutoApprovalDelayInSeconds
    # 11	Expiration
    # 12	NumberOfSimilarHITs
    # 13	LifetimeInSeconds
    # 14	AssignmentId
    # 15	WorkerId **** THIS ****
    # 16	AssignmentStatus
    # 17	AcceptTime
    # 18	SubmitTime
    # 19	AutoApprovalTime
    # 20	ApprovalTime
    # 21	RejectionTime
    # 22	RequesterFeedback
    # 23	WorkTimeInSeconds
    # 24	LifetimeApprovalRate
    # 25	Last30DaysApprovalRate
    # 26	Last7DaysApprovalRate
    # 27	Input.opt1 **** THIS ****
    # 28	Input.opt2 **** THIS ****
    # 29	Input.type **** THIS ****
    # 30	Input.corr_opt **** THIS ****
    # 31	Input.sent **** THIS ****
    # 32	Input.full_id **** THIS ****
    # 33	Answer.Answer **** THIS ****
    # 34	Approve
    # 35	Reject
    
    $worker = $F[15];
    $options[0] = $F[27];
    $options[1] = $F[28];
    $type = $F[29];
    $correct_index = $F[30] -1;
    $sentence = $F[31];
    $id = $F[32];
    $response = $F[33];
    
    if ($type eq "foil") {
	next;
    }

    $answer_status = "F";
    $correct_form = $options[$correct_index];
    if ($response eq $correct_form) {
	$answer_status = "T";
    }

    print join ("\t",
	($worker,$type,$answer_status,$response,$correct_form,$options[0],$options[1],$id,$sentence)),
	"\n";
}

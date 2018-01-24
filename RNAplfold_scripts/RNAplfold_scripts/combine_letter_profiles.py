#!/usr/bin/env/python

import sys

def list_to_str(lst):
    ''' Given a list, return the string of that list with tab separators 
    '''
    return reduce( (lambda s, f: s + '\t' + str(f)), lst, '')


fEprofile = open(sys.argv[1])
Eprofiles = fEprofile.readlines()

fHprofile = open(sys.argv[2])
Hprofiles = fHprofile.readlines()

fIprofile = open(sys.argv[3])
Iprofiles = fIprofile.readlines()

fMprofile = open(sys.argv[4])
Mprofiles = fMprofile.readlines()

mw = int(sys.argv[5])

fhout = open(sys.argv[6], 'w')

for i in range(0, len(Eprofiles)/2):
	id = Eprofiles[i*2].split()[0]
	print >>fhout, id
	H_prob =  Hprofiles[i*2+1].split()
	I_prob =  Iprofiles[i*2+1].split()
	M_prob =  Mprofiles[i*2+1].split()
	E_prob =  Eprofiles[i*2+1].split()			
	P_prob = map( (lambda a, b, c, d: 1-float(a)-float(b)-float(c)-float(d)), H_prob, I_prob, M_prob, E_prob)
	print >>fhout, list_to_str(P_prob[mw-1:len(P_prob)])
	print >>fhout, list_to_str(H_prob[mw-1:len(P_prob)])
	print >>fhout, list_to_str(I_prob[mw-1:len(P_prob)])
	print >>fhout, list_to_str(M_prob[mw-1:len(P_prob)])
	print >>fhout, list_to_str(E_prob[mw-1:len(P_prob)])



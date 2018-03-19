from igraph import*

file = open("text8", "r")
input = file.read()
input = input.replace("/n", "")
input =  input.split()
wordlist = [x.lower() for x in input]
wlen = len(wordlist)

#print wlen

#for p in wordlist: print p

wordGraph = Graph()
wordGraph.add_vertices(len(wordlist))
wordGraph.vs["name"] = wordlist
wordGraph.vs["label"]=wordlist

for idx, word in enumerate(wordlist):
    if idx == 0:
        wordGraph.add_edges([(word, wordlist[idx+1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+1])
        wordGraph.add_edges([(word, wordlist[idx+2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+2])
    elif idx == 1:
        wordGraph.add_edges([(word, wordlist[idx-1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-1])
        wordGraph.add_edges([(word, wordlist[idx+1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+1])
        wordGraph.add_edges([(word, wordlist[idx+2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+2])
    elif idx == (wlen-2):
        wordGraph.add_edges([(word, wordlist[idx-2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-2])
        wordGraph.add_edges([(word, wordlist[idx-1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-1])
        wordGraph.add_edges([(word, wordlist[idx+1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+1])
    elif idx == (wlen-1):
        wordGraph.add_edges([(word, wordlist[idx-2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-2])
        wordGraph.add_edges([(word, wordlist[idx-1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-1])
    else:
        wordGraph.add_edges([(word, wordlist[idx-2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-2])
        wordGraph.add_edges([(word, wordlist[idx-1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx-1])
        wordGraph.add_edges([(word, wordlist[idx+1])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+1])
        wordGraph.add_edges([(word, wordlist[idx+2])])
        print 'Added edges between %s and %s' % (word, wordlist[idx+2])

wordGraph=wordGraph.simplify()

print wordGraph

layout = wordGraph.layout_lgl()
#layout = wordGraph.layout_kamada_kawai()

out = plot(wordGraph, layout = layout)

out.save(name + '_testgraph.svg')

package org.aksw.limes.core.measures.measure.semantic.edgecounting.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Queue;

import org.aksw.limes.core.measures.measure.semantic.edgecounting.dictionary.SemanticDictionary;

import edu.mit.jwi.item.ISynset;
import edu.mit.jwi.item.ISynsetID;
import edu.mit.jwi.item.Pointer;

public class ShortestPathFinder {

    public static long duration = 0l;
    
    public static int shortestPath(ArrayList<ArrayList<ISynsetID>> synset1Tree,
            ArrayList<ArrayList<ISynsetID>> synset2Tree) {
        
        long begin = System.currentTimeMillis();
        
        int path = Integer.MAX_VALUE;

        if (synset1Tree == null || synset2Tree == null)
            return -1;

        if (synset1Tree.isEmpty() == true || synset2Tree.isEmpty() == true)
            return -1;

        int path1Pos, path2Pos;

        for (List<ISynsetID> synset1HypernymPath : synset1Tree) {
            /*
             * System.out.println("1"); for(ISynsetID id1: synset1HypernymPath){
             * System.out.println(""+id1.toString()); }
             */
            for (List<ISynsetID> synset2HypernymPath : synset2Tree) {

                /*
                 * System.out.println("2"); for(ISynsetID id2:
                 * synset2HypernymPath){ System.out.println(""+id2.toString());
                 * }
                 */

                path1Pos = 0;
                path2Pos = 0;
                while ((path1Pos < synset1HypernymPath.size()) && (path2Pos < synset2HypernymPath.size())
                        && (synset1HypernymPath.get(path1Pos).getOffset() == synset2HypernymPath.get(path2Pos)
                                .getOffset())) {
                    ++path1Pos;
                    ++path2Pos;
                }
                int newPath = synset1HypernymPath.size() + synset2HypernymPath.size() - 2 * path1Pos;
                if (newPath < path) {
                    path = newPath;
                }

            }
        }
        if (path == Integer.MAX_VALUE)
            path = -1;
        
        long end = System.currentTimeMillis();
        duration = end - begin;
        return path;

    }

    public static int shortestPath(ISynset synsetID1, ISynset synsetID2, SemanticDictionary dictionary) {
        long begin = System.currentTimeMillis();
        
        int length = 0;

        if (synsetID1 == null || synsetID2 == null || dictionary == null)
            return -1;

        Queue<ISynsetID> queue1 = new LinkedList<ISynsetID>();
        queue1.add(synsetID1.getID());
        Queue<ISynsetID> queue2 = new LinkedList<ISynsetID>();
        queue2.add(synsetID2.getID());

        // keep track of the visited nodes and their length from the synset1
        LinkedHashMap<ISynsetID, Integer> visited1 = new LinkedHashMap<ISynsetID, Integer>();
        LinkedHashMap<ISynsetID, Integer> noHypers1 = new LinkedHashMap<ISynsetID, Integer>();

        // keep track of the visited nodes and their length from the synset2
        LinkedHashMap<ISynsetID, Integer> visited2 = new LinkedHashMap<ISynsetID, Integer>();
        LinkedHashMap<ISynsetID, Integer> noHypers2 = new LinkedHashMap<ISynsetID, Integer>();

        int length1 = 0;
        int length2 = 0;
        int minLength = Integer.MAX_VALUE;

        visited1.put(synsetID1.getID(), length1);
        visited2.put(synsetID2.getID(), length2);

        while (!queue1.isEmpty() || !queue2.isEmpty()) {

            if (!queue1.isEmpty()) {

                /*
                 * System.out.println("Visited1"); for (Entry<ISynsetID,
                 * Integer> entry : visited1.entrySet()) { ISynsetID key =
                 * entry.getKey(); Integer value = entry.getValue();
                 * System.out.println("key "+key+" "+value); }
                 */

                ISynsetID id = queue1.poll();

                /*
                 * System.out.println("1 Poping: "+id.toString());
                 * System.out.println("-----------------");
                 */
                if (visited2.containsKey(id)) {
                    length = visited1.get(id) + visited2.get(id);
                    if (length < minLength)
                        minLength = length;
                    // return length;
                }
                length1 = visited1.get(id) + 1;
                ISynset synset = dictionary.getDictionary().getSynset(id);

                List<ISynsetID> hypernymIds = synset.getRelatedSynsets(Pointer.HYPERNYM);

                for (ISynsetID hyperID : hypernymIds) {
                    if (!visited1.containsKey(hyperID)) {
                        visited1.put(hyperID, length1);
                        queue1.add(hyperID);
                    }
                }

                List<ISynsetID> instanceHypernymIds = synset.getRelatedSynsets(Pointer.HYPERNYM_INSTANCE);
                for (ISynsetID hyperID : instanceHypernymIds) {
                    if (!visited1.containsKey(hyperID)) {
                        visited1.put(hyperID, length1);
                        queue1.add(hyperID);
                    }
                }
                if (synset.getType() != 1) {
                    if (hypernymIds.isEmpty() && instanceHypernymIds.isEmpty()) {
                        noHypers1.put(id, visited1.get(id));
                    }
                }

            }

            if (!queue2.isEmpty()) {

                /*
                 * System.out.println("Visited2"); for (Entry<ISynsetID,
                 * Integer> entry : visited2.entrySet()) { ISynsetID key =
                 * entry.getKey(); Integer value = entry.getValue();
                 * System.out.println("key "+key+" "+value); }
                 */

                ISynsetID id = queue2.poll();

                /*
                 * System.out.println("2 Poping: "+id.toString());
                 * System.out.println("-----------------");
                 */
                if (visited1.containsKey(id)) {
                    length = visited1.get(id) + visited2.get(id);
                    if (length < minLength)
                        minLength = length;
                    // return length;
                }
                length2 = visited2.get(id) + 1;
                ISynset synset = dictionary.getDictionary().getSynset(id);

                List<ISynsetID> hypernymIds = synset.getRelatedSynsets(Pointer.HYPERNYM);
                for (ISynsetID hyperID : hypernymIds) {
                    if (!visited2.containsKey(hyperID)) {
                        visited2.put(hyperID, length2);
                        queue2.add(hyperID);
                    }
                }

                List<ISynsetID> instanceHypernymIds = synset.getRelatedSynsets(Pointer.HYPERNYM_INSTANCE);
                for (ISynsetID hyperID : instanceHypernymIds) {
                    if (!visited2.containsKey(hyperID)) {
                        visited2.put(hyperID, length2);
                        queue2.add(hyperID);
                    }
                }

                if (synset.getType() != 1) {
                    if (hypernymIds.isEmpty() && instanceHypernymIds.isEmpty()) {
                        noHypers2.put(id, visited2.get(id));
                    }
                }

            }

        }
        //

        if (minLength == Integer.MAX_VALUE) {
            Entry<ISynsetID, Integer> min1 = Collections.min(noHypers1.entrySet(),
                    Comparator.comparing(Entry::getValue));
            Entry<ISynsetID, Integer> min2 = Collections.min(noHypers2.entrySet(),
                    Comparator.comparing(Entry::getValue));
            minLength = min1.getValue() + min2.getValue() + 2;
        }
        
        long end = System.currentTimeMillis();
        duration = end - begin;
        return minLength;
    }

}

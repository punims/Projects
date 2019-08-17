


package oop.ex4.data_structures;


/** Imports */

import java.lang.Iterable;
import java.util.Iterator;
import java.util.ArrayList;


/**
 * Created by edanp on 5/11/2017.
 */
public  class AvlTree implements Iterable<Integer> {


    /**
     * Constants and Declarations
     */
     ArrayList<Integer> iteratorList;
     Node root;
     int size;
     private static int NOT_FOUND = -1;
     private static int STARTING_DEPTH = 0;
     private static int STARTING_SIZE = 0;
     private static int TOO_MANY_LEFT = 2;
     private static int TOO_MANY_RIGHT = -2;
     private static int NEGATIVE = 0;

    /**
     * Constructors
     */

    /**
     * The basic constructor
     */
    public AvlTree() {
        root = null;
        size = STARTING_SIZE;
        iteratorList = new ArrayList<Integer>();
    }

    /**
     * An avltree constructor that takes an array of numbers and creates a tree with those numbers
     * @param data
     */
    public AvlTree(int[] data){
        root = null;
        size = STARTING_SIZE;
        iteratorList = new ArrayList<Integer>();
        if(data != null)
        for (int item: data) add(item);
        }



    /**
     * a constructor that takes an existing avltree tree and creates a new one with the same nodes but not
     * necessarily in the same  order. Basically a "deep copy".
     * @param avltree
     */
    public AvlTree(AvlTree avltree){

        root = null;
        size = STARTING_SIZE;
        iteratorList = new ArrayList<Integer>();
        if(avltree != null)
            for (int item: avltree) add(item);
    }

    /**
     * returns an iterator, must be  implemented
     */
    public Iterator <Integer> iterator(){
        if(root == null) return null;
        // must be implemented to use the interface Iterable
        buildArrayList(root);
        return iteratorList.iterator();
    }

    /**
     * A method for building an ArrayList which contains all the nodes in a rising order
     * adds all the data of the tree recursively
     */

    public void buildArrayList(Node currentNode){
        if(currentNode == null){
            return;
        }
        buildArrayList(currentNode.left);
        iteratorList.add(currentNode.data);
        buildArrayList(currentNode.right);
    }


    /**
     * Methods
     */


    /**
     * a method for adding a  new node to the tree while making sure it is kept balanced.
     * @param newValue a new number
     * @return true if the value was added successfully, false otherwise.
     */
    public boolean add(int newValue) {
        Node newNode = new Node(newValue);

        if (contains(newValue) != NOT_FOUND)
            return false;
        addHelper(newNode); // a new node has been added to the tree so the tree size also grows.
        size++;
        treeBalancer(newNode); // check if the tree is balanced and balance if  needed
        return true;
    }

        // the first if condition wasn't met so the root  has a value.


    /**
     * a helping method for adding nodes to the tree, same as adding to any BST
     * @param newNode
     */
    private void addHelper(Node newNode) {
        if(root == null){
            root = newNode;
        } else {
            Node focusNode = root;
            Node parent;
            while (true) {
                parent = focusNode;
                if (newNode.getData() < focusNode.getData()) {
                    focusNode = focusNode.getLeftChild();
                    if (focusNode == null) {
                        parent.left = newNode;
                        newNode.getParent(newNode, root);
                        return;
                    }
                    continue;
                } else {
                    focusNode = focusNode.getRightChild();
                    if (focusNode == null) {
                        parent.right = newNode;
                        newNode.getParent(newNode, root);
                        return;
                    }
                    continue;
                }
            }
        }
    }


    /**
     * a helping recursive function which checks if a tree is balanced from a given node up to the root.
     * note that a negative node balance means there are more nodes to the right and a positive node balance means
     * there are more nodes to the left
     * @param currentNode
     */
    private void treeBalancer(Node currentNode) {
        if(currentNode == null){
            return; // don't act on a null Node.
        }
        currentNode.getHeight(); // update the height
        int currentBalance = currentNode.getBalance(); // update the balance
        if (currentBalance <= TOO_MANY_RIGHT) {
            if (currentNode.right.getBalance() > NEGATIVE) {// a Right Left case
                rightLeftRotation(currentNode);
            } else
                leftLeftRotation(currentNode); // else its a Right Right case and so needs to be rotated left
        }
        else if(currentBalance >= TOO_MANY_LEFT){
            if(currentNode.left.getBalance() < NEGATIVE){
                leftRightRotation(currentNode); // its a Left Right case
            }else{
                rightRightRotation(currentNode); // its a Left Left case and so needs to be rotated right
            }
        }
        if(currentNode.parent == null){ // we've checked up to the root so we can safely say the tree is balanced.
            return;
        }
        treeBalancer(currentNode.parent);
        // if the current Node isn't the root check if the Parent is balanced or is
        // the root.
    }

    /**
     * Right Right case that moves all nodes to the left to  balance the tree.
     */
    public void leftLeftRotation(Node localRoot) {
        Node newRoot = localRoot.right;
        localRoot.right = newRoot.left;
        if(localRoot.right != null){
            localRoot.right.getParent(localRoot.right,root);
        }
        newRoot.left = localRoot;
        if(localRoot == root) {
            root = newRoot;//if the newRoot is the actual new root of the entire tree it must be changed.
        }else{
            if(localRoot == localRoot.parent.right){ // decide if to connect to right or left
            localRoot.parent.right = newRoot;}
            else{
                localRoot.parent.left = newRoot;
            }
        }
        localRoot.getParent(localRoot, root);
        newRoot.getParent(newRoot, root); // all new parents must be assigned
    }

    /**
     * a Left Left case the moves all the nodes to the right to balance the tree
     */
    public void rightRightRotation(Node localRoot) {
        Node newRoot = localRoot.left;
        localRoot.left = newRoot.right;
        if(localRoot.left != null){
            localRoot.left.getParent(localRoot.left, root);
        }
        newRoot.right = localRoot;
        if(localRoot == root){
            root = newRoot;
        } else {
            if(localRoot == localRoot.parent.right){ // decide if to connect to right or left
            localRoot.parent.right = newRoot;}// updating the old child of the original root.
            else{
                localRoot.parent.left = newRoot;
            }
        }
        localRoot.getParent(localRoot, root); // updating the new parent
        newRoot.getParent(newRoot, root);
    }

    /**
     * A Right Left case using the already made functions
     */

    public void rightLeftRotation(Node localRoot) {
        rightRightRotation(localRoot.right);
        leftLeftRotation(localRoot);

    }

    /**
     * A Left Right case using the already made functions
     */

    public void leftRightRotation(Node localRoot) {
        leftLeftRotation(localRoot.left);
        rightRightRotation(localRoot);
    }

    /**
     * A function that checks if the tree contains a certain value or not
     * returns -1 if the value wasnt found, otherwise return the depth(how far away from the root) of the node.
     */

    public int contains(int searchVal){
        Node current = root;
        int depth = STARTING_DEPTH; // 0
        while(current != null){ //there will always be a point when we reach a null Node
            if(searchVal ==  current.data)
                return depth; //we  found the value  and so its returned
            if(searchVal < current.data){ //checks  left where  the values are lower
                current = current.left;
                depth ++;
                continue;
            }else{
                current = current.right; // checks right where the values are higher.
                depth ++;
                continue;
            }
        }
    return NOT_FOUND; // the node is null so the value is not in the tree.
    }

    /**
     * a method for returning the size of the tree (number of nodes in the tree)
     */
    public int size(){
        return this.size;
        // been keeping track of the size during the add and delete phase with a starting size of 0.
    }

    /**
     * a method for removing a single node with the given value if it exists in the tree.
     * @param toDelete
     * @return true if removed, false otherwise
     */
    public boolean delete(int toDelete){
        if (contains(toDelete) == NOT_FOUND)
            return false;
        Node forBalance = searchAndDelete(toDelete);
        // deletes the node and returns a node from which we check the balance.
        size --;
        treeBalancer(forBalance);
        return true;
    }

    /**
     * a normal BST deletion method which returns the node from which we need to check the balance.
     * @param toDelete
     * @return the node from  which balance needs to be checked.
     */
    public Node searchAndDelete(int toDelete){
        // we already know the node is contained, so we'll search for it
        Node removedNode = searchForNode(toDelete); // a placeholder  for the node
        Node toBeReturned = removedNode.parent;
        if(removedNode.left == null && removedNode.right == null){  // no children scenerio:
            if (toBeReturned == null){ // the deleted Node was the root with no children.
                root  = null;
                return null;
            }
            if(toBeReturned.left == removedNode){ //checking if our node is left or right to its' parent
                toBeReturned.left = null;
                return toBeReturned;
            }else{
                toBeReturned.right = null;
                return toBeReturned;
            }
        }
        boolean leftCaseSon = (removedNode.left != null && removedNode.right == null); // only one son case.
        boolean rightCaseSon = (removedNode.left == null && removedNode.right != null);
        if(leftCaseSon){ //if there's only a left son, connect it to the removed nodes parent
            if(removedNode == root){
                root = removedNode.left;
                root.parent = null;
                return root; // in the case this is the root just assign the node as the new root.
            }
            if(toBeReturned.left == removedNode){
            toBeReturned.left = removedNode.left;
            toBeReturned.left.parent = toBeReturned;}
            else{
                toBeReturned.right =removedNode.left;
                toBeReturned.right.parent = toBeReturned;
            }
            return toBeReturned;
        }
        if(rightCaseSon){
            if(removedNode == root){
               root = removedNode.right;
                root.parent = null;
                return root;}
            if(toBeReturned.right == removedNode){
            toBeReturned.right =  removedNode.right;
            toBeReturned.right.parent = toBeReturned;
            }// assign new parent depending on original placement
            else{
                toBeReturned.left = removedNode.right;
                toBeReturned.left.parent = toBeReturned;
                }
            return toBeReturned;
        }
        if(removedNode.right != null && removedNode.left != null){ // in case there are two sons for the removed node
            return twoSonRemoval(removedNode); // the replacement is the minimal node of the right subtree.
        }
        return null; // won't reach  this part of the function
    }

    /**
     * a Helping function for deleting a Node with two children
     * we'll search for the minimum Node and replace it with the node to be deleted if the min value has  a child
     * we'll make a small rotation. after finishing this process the deleted node will switch places with the min
     * value and will be deleted normally as a leaf.
     */

    public Node twoSonRemoval(Node removedNode){
        Node replacementNode = searchMinimalNode(removedNode.right); // search for the min node in the right subtree.
        if(replacementNode.right != null){  //the replacement node has a child
            leftLeftRotation(replacementNode);  //the minimal node will be rotated left.
        }
        Node originalParent = replacementNode.parent; // note that it's made after the potential left rotation
        Node originalRoot = root;
        if(removedNode == root){ //in case we're removing the root
            replacementNode.parent = null;
            root = replacementNode;
        }else{
            replacementNode.parent = removedNode.parent;
            if(removedNode.parent.left == removedNode){ // find if the removed node was connected from the left or right
                removedNode.parent.left = replacementNode;
            }else{
                removedNode.parent.right = replacementNode;
            }
        }
        if(removedNode.left == replacementNode){ //in the case we're  switching a parent with its' first child
            replacementNode.left = null;
        }else{
        replacementNode.left = removedNode.left;
        replacementNode.left.getParent(replacementNode.left, root);}
        if(removedNode.right  == replacementNode){
            replacementNode.right = null;
        }else{
        replacementNode.right = removedNode.right;
        replacementNode.right.getParent(replacementNode.right, root);
        }
        originalParent.left = null; // remove the  node from the original parent
        // up to here the node has been successfully replaced.
        if(originalParent == originalRoot) {
            return root; // root was replaced and was the parent, return the new root.
        }
        if(originalParent == removedNode){ // in case the parent was what we removed its' not in the tree
            return replacementNode;
        }
        return originalParent;
        // returns the OG parent so we can balance the tree.
    }


    /**
     * a Helping function to search for a minimal value of a subtree (always go to the left child)
     */

    public Node searchMinimalNode(Node startingNode){
        Node current = startingNode;
        while(current.left != null){
            current = current.left;
            continue; // repeat going down the left child tree until you hit a null node
        }
        return current; // we've reached the minimal node so it's returned.
    }

    /**
     * a BST searching method
     */
    public Node searchForNode(int searchValue){
        Node current = root;
        while(current != null && root.data != searchValue){ //there will always be a point when we reach a null Node
            if(searchValue ==  current.data)
                return current; //we  found the value  and so its returned
            if(searchValue < current.data){ //checks  left where  the values are lower
                current = current.left;
                continue;
            }else{
                current = current.right; // checks right where the values are higher.
                continue;
            }
        }
        return root; // we searched and found the root which is why we never enter the while loop
    }


    /**
     * Static methods:
     */

    /**
     * Calculates the minimum number of nodes in an AVL tree of height h.
     */
    public static int findMinNodes(int h){
        if (h == 0) return 1;
        if (h == 1) return 2;
        int oneLevelDown= 2, twoLevelDown = 1, result = 0;
        for(int i = 0; i < h-1; i++) {
            result = oneLevelDown + twoLevelDown + 1; // the algorithm for calculation.
            twoLevelDown = oneLevelDown;
            oneLevelDown = result;
        }
        return result;
    }

    /**
     * Calculates the maximum number of nodes in an AVL tree of height h.
     */
    public static int findMaxNodes(int h) {
        return (int)Math.pow(2,h+1) - 1; // a full BST
    }
}



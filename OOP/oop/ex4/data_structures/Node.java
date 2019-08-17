/**
 * Created by edanp on 5/11/2017.
 */


package oop.ex4.data_structures;


public class Node {


    /**
     * Constants
     */

    private static final int BEGINNING_HEIGHT = -1;
    private static final int BEGINNING_BALANCE = 0;
    private static final int BALANCE_INCREASE = 1;
    private static final int HEIGHT_INCREASE = 1;
    private static final int NO_CHILDREN = 0;
    Node left, right, parent;
    int height, data, balance;



    /**
     * Constructor for the Node object
     */

    public Node(int key){
        parent = null;
        right = null;
        left = null;
        balance = BEGINNING_BALANCE;
        data = key;
        height = BEGINNING_HEIGHT;

        //each Node contains two connecting nodes (as this will be a tree), data and its height in the tree
        // the methods for calculating the height will be within the AvlTree class.

    }

    /** Node Methods:
     */


    /**
     * A method for getting the data out of a Node object
     * @return int data.
     */
    public int getData(){
        return data;
    }


    /**
     * A method for getting the left node
     */
    public Node getLeftChild(){
        return left;
    }


    /**
     * A method for getting the right node
     */
    public Node getRightChild(){
        return right;
    }

    /**
     * A method for getting the Node balance
     */

    public int getBalance(){
        if(this.right == null && this.left == null){
            balance = NO_CHILDREN; //if the node is a leaf its' height is zero
            return balance;
        }
        if(this.right == null){ // only right is null
            balance = left.getHeight() + BALANCE_INCREASE;
            return balance;}
        else if(this.left == null){ // only left is null.
            balance = -(right.getHeight() + BALANCE_INCREASE); //the right is always negative
            return balance;
        }
        // reaching here means both left and right have values
        balance = this.left.getHeight() - this.right.getHeight();
        return balance;
    }

    /**
     * A method for getting the height of a Node object within an AvlTree
     */
    public int getHeight(){
        heightUpdater();
        return height;
    }

    /**
     * a recursive method for getting and assigning the parent of a node in the avl tree.
     */
    public Node getParent(Node node, Node root){

        Node previousNode = null; //if we're looking for the parent of the root null is returned
        //commence a binary search for the node while always remembering the previous node.
        while(node.data != root.data){
            if(node.data > root.data){
                previousNode = root;
                root = root.right;
                continue;
            }
            if(node.data < root.data){
                previousNode = root;
                root = root.left;
                continue;
            }
        }
        //we've reached this point which means we've found the  node
        node.parent = previousNode;
        return node.parent;

        }



    /**
     * A recursive  function that helps the getHeight function
     * by updating the heights for all of the nodes up to the root in question.
     */
    public void heightUpdater(){
        if(this.right == null && this.left == null){
            height = NO_CHILDREN; //if the node is a leaf its' height is zero
        }
        else if(this.right == null){
            height = left.getHeight() +HEIGHT_INCREASE; // if only right is null
        }
        else if(this.left == null){
            height = right.getHeight() +HEIGHT_INCREASE; // if only left is null
        }
        else{
            height = Math.max(left.getHeight(), right.getHeight()) +HEIGHT_INCREASE;
            // if the Node has children it will recursively look for their values
            // as a sidenote if a node has only one child to one side the side with no child has a height of -1
        }
    }
}

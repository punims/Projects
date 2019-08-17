package oop.ex6.Block;

import oop.ex6.Variables.Variable;

import java.util.HashMap;


public abstract class Block {

    private Block parent;
    private HashMap<String, Variable> localVariables;

    /**
     * The normal format constructor for any type of block.
     *
     * @param parent
     */
    public Block(Block parent) {
        this.parent = parent;
        this.localVariables = new HashMap();
    }

    /**
     * A constructor made for creating the Global block. (does not have a parent value)
     */
    public Block() {
        this.parent = null;
        this.localVariables = new HashMap();
    }

    /**
     * A method which adds a variable to the block's HashMap if it does not exist within the HashMap
     * and returns true, else it will return false.
     *
     * @param variable
     */
    public boolean addLocalVariable(Variable variable) {
        if(variable == null){
            return false;
        }
        if (!localContains(variable)) {
            localVariables.put(variable.getName(), variable);
            return true;
        } else
            return false;
    }

    /**
     * A method which asks if a value is held within the local Hashmap.
     *
     * @param variable
     * @return
     */
    public boolean localContains(Variable variable) {
        return this.localVariables.containsKey(variable.getName());
    }

    /**
     * A method which asks if a value is held within the local Hashmap.
     *
     * @param variableName
     * @return
     */

    // local contains
    public boolean localContains(String variableName) {
        return this.localVariables.containsKey(variableName);
    }

    /**
     * method which checks if a variable is contained in one of the outer scopes including the current one.
     * @param variableName
     * @return
     */
    public boolean globalContains(String variableName) {
        Block currentBlock = this;
        while(currentBlock != null){
            if(currentBlock.localContains(variableName)){
                return true;
            } currentBlock = currentBlock.parent;
        } return false;
    }

    /**
     * Overridden method with different parameters. Created for future possible use.
     * @param variable
     * @return
     */
    public boolean globalContains(Variable variable) {
        Block currentBlock = this;
        while (currentBlock != null) {
            if (localContains(variable)) {
                return true;
            }
            currentBlock = currentBlock.parent;
        }
        return false;
    }


    /**
     * a getter function for a variable in the Hashmap
     * @param variableName
     * @return
     */
    public Variable getVariable(String variableName) {
        return this.localVariables.get(variableName);
    }

    /**
     * A getter function for one of the global variables starting with the current block.
     */
    public Variable getGlobalVariable(String variableName){
        Block currentBlock = this;
        while(currentBlock != null) {
            if (currentBlock.localContains(variableName)) {
                return currentBlock.getVariable(variableName);
            } currentBlock = currentBlock.parent;
        }
        return null;
        // unreachable as this function is only used if we've checked that the variable exists.
    }

    /**
     * returns the local variable Hashmap.
     *
     * @return
     */
    public HashMap getLocalVariables() {
        return localVariables;
    }


    /**
     * A function which gets a string of the variables' class using the variable name.
     */
    public String getVariableType(String variableName) {
        return getGlobalVariable(variableName).getClass().getName();
    }

    /**
     * getter function for a block's parent.
     * @return
     */
    public Block getParent(){
        return parent;
    }
}

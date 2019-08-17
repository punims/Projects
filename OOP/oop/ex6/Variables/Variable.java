package oop.ex6.Variables;

import oop.ex6.Block.*;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public abstract class Variable {

    public static final
    String S_CHAR = "oop.ex6.Variables.SChar", S_BOOLEAN = "oop.ex6.Variables.SBoolean",
            S_DOUBLE = "oop.ex6.Variables.SDouble", S_INT = "oop.ex6.Variables.SInt",
            S_STRING = "oop.ex6.Variables.SString";
    static final String VARIABLE_NAME = "_[\\w]+|[A-Za-z]+[\\w]*";

    static Pattern p;
    static Matcher m;

    String name;
    boolean assigned;
    boolean isFinal;
    Block currentBlock;

    public Variable() {
    }
    //default constructor

    public Variable(String name, boolean isFinal, boolean assignment, Block currentBlock) {
        this.name = name;
        assigned = assignment;
        this.isFinal = isFinal;
        this.currentBlock = currentBlock;

    }

//    boolean checkName(){
//    }

    /**
     * a function which tries to assign a value to a variable.
     * checks to see if the value is another variable if it exists in one of the scopes.
     *
     * @param value
     * @return
     */
    public boolean assignValue(String value) {

        if (checkValue(value)) {
            this.assigned = true;
            return true;
        }
        return false;
    }


    /**
     * abstract function which checks a value of a possible variable value.
     * @param value
     * @return
     */
    abstract boolean checkValue(String value);


    /**
     * is final getter.
     * @return
     */
    public boolean isFinal() {
        return isFinal;
    }


    /**
     * is assigned getter.
     * @return
     */
    public boolean isAssigned(){return assigned;}

    /**
     * a variable name getter.
     * @return
     */
    public String getName() {
        return this.name;
    }


    /**
     * @param first
     * @param second
     * @return true if the assignment is legal according to the class type. false otherwise.
     */
    public static boolean assignable(String first, String second) {
        switch (first) {
            case S_BOOLEAN:
                return (second.equals(S_BOOLEAN) || second.equals(S_INT) || second.equals(S_DOUBLE));
            case S_CHAR:
                return (second.equals(S_CHAR));
            case S_DOUBLE:
                return (second.equals(S_INT) || second.equals(S_DOUBLE));
            case S_INT:
                return (second.equals(S_INT));
            default:
                return second.equals(S_STRING);
        }
    }

    /**
     * a helping  function which checks if a final variable has a set value.
     *
     * @param isFinal
     * @param assigned
     * @return true if the variable does not defy this rule, false if the rule is broken.
     */
    public static boolean finalHasValue(boolean isFinal, boolean assigned) {
        if (isFinal)
            if (!assigned) {
                return false;
            }
        return true;
    }

    /**
     * getter function which gets a variable's class name.
     */
    public String variableGetTypeName(){
        return getClass().getName();
    }
}

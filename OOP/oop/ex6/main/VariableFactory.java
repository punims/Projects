package oop.ex6.main;

import oop.ex6.Variables.*;
import oop.ex6.Block.*;


public class VariableFactory {

    final public static int STRING = 11, CHAR = 12, DOUBLE = 13, BOOLEAN = 14, INT = 15;
    final static String NO_VALUE = "";


    /**
     * A factory method which creates a new Variable object according to its' type and also states whether or not
     * it has a value and whether or not it is final.
     *
     * @param type
     * @param name
     * @param value
     * @param isFinal
     * @return
     */
    public static Variable createVariable(int type, String name, String value, boolean isFinal, Block currentBlock) {
        boolean hasValue = (!value.equals(NO_VALUE));
        Variable nullVariable = null;
        if (valueIsVariable(value)) {
            if (!currentBlock.globalContains(value)) {
                return nullVariable;
                // the value was a variable which was not defined.
            }
            Variable variableCheck = currentBlock.getGlobalVariable(value);
            if (!declarationAssignable(type, variableCheck) ||  !variableCheck.isAssigned()) {
                return nullVariable;
            }
            // check if the variable type matches or if its assigned.
        }
        if (!Variable.finalHasValue(isFinal, hasValue))
            // if the variable is final but has no value.
            return nullVariable;
        switch (type) {
            case STRING:
                return new SString(name, isFinal, hasValue, currentBlock);
            case BOOLEAN:
                return new SBoolean(name, isFinal, hasValue, currentBlock);
            case CHAR:
                return new SChar(name, isFinal, hasValue, currentBlock);
            case INT:
                return new SInt(name, isFinal, hasValue, currentBlock);
            case DOUBLE:
                return new SDouble(name, isFinal, hasValue, currentBlock);
        }
        return null;
        //unreachable.
    }

    /**
     * a method which decides if a value is a possible Variable name.
     *
     * @param value
     * @return
     */
    private static boolean valueIsVariable(String value) {
        if (value.equals("true") | value.equals("false")) {
            return false;
            // true and false look like variables according to regex, but cannot be variables.
        }
        return value.matches(FirstScan.VARIABLE_NAME);
    }

    /**
     * a method which checks if a variable has the right type to be assigned to a declared variable.
     *
     * @return
     */
    public static boolean declarationAssignable(int type, Variable variableCheck) {
        String variableType = variableCheck.getClass().getName();
        switch (type) {
            case STRING:
                if (variableType.equals(Variable.S_STRING))
                    return true;
            case CHAR:
                if (variableType.equals(Variable.S_CHAR))
                    return true;
            case BOOLEAN:
                if (variableType.equals(Variable.S_BOOLEAN))
                    return true;
            case INT:
                if (variableType.equals(Variable.S_INT))
                    return true;
            case DOUBLE:
                if (variableType.equals(Variable.S_DOUBLE))
                    return true;
        }
        return false;
    }
}


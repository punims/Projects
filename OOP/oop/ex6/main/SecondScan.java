package oop.ex6.main;

import oop.ex6.Block.*;

import java.util.ArrayList;
import java.util.regex.Pattern;

import oop.ex6.Exceptions.BlockException;
import oop.ex6.Exceptions.IllegalLineException;
import oop.ex6.Exceptions.VariableException;
import oop.ex6.Exceptions.WrongMethodParameters;
import oop.ex6.Variables.*;

/**
 * Created by edanp on 6/19/2017.
 */
public class SecondScan extends FirstScan {


    private final static String ERROR = "error";
    private final static int METHOD_NAME = 2, METHOD_CALL = 0, METHOD_VARIABLES = 1;

    private int blockCounter, codeType, arrayCounter;
    private Block currentBlock;
    private Method theMethod;
    private ArrayList<String> lines;


    /**
     * Constructor for the second scan object.
     * @param scannedLines
     */
    public SecondScan(ArrayList<String> scannedLines) {
        blockCounter = 0;
        codeType = 0;
        arrayCounter = 0;
        currentBlock = Global.getGlobal();
        theMethod = null;
        lines = scannedLines;
        this.global = Global.getGlobal();
    }


    /**
     * a Method which scans the code once more only with an updated global. Now all that's left to search for
     * are illogical or illegal declarations within Blocks.
     *
     * @param lines
     */
    public void secondLook(ArrayList<String> lines) throws BlockException, VariableException,
            IllegalLineException {

        blockCounter = 0;
        codeType = 0;
        arrayCounter = 0;
        currentBlock = global;
        String firstVariable = null, secondVariable = null;
        theMethod = null;


        for (String line : lines) {
            codeType = getCodeType(line);
            if (codeType == METHOD_DECLARATION_NUM) {
                blockCounter++;
                p = Pattern.compile(METHOD_DECLARATION);
                m = p.matcher(line);
                m.matches();
                String methodName = m.group(METHOD_NAME);
                currentBlock = global.getMethod(methodName);
            }
            if (blockCounter > 0) {
                if (codeType == VARIABLE_ASSIGNMENT_NUM) {
                    if (!variableAssignmentChecker(line))
                        throw new VariableException();
                }
                if (codeType == VARIABLE_DECLARATION_NUM) {
                    createVariables(line);
                }
                if (codeType == IF_WHILE_NUM) {
                    blockCounter++;
                    currentBlock = BlockFactory.createIfWhile(currentBlock, line);
                }
                if (codeType == METHOD_CALL_NUM) {
                    methodCallCheck(line, firstVariable, secondVariable);
                }
                if (codeType == CURLY_BRACKET_NUM) {
                    blockCounter--;
                    currentBlock = currentBlock.getParent();
                }

            }
        }

    }


    /**
     * a method which checks if a variable is assigned only made for the second reach through
     * designed almost the same as the first with the only exception of checking Variable existance on a
     * Global scope.
     *
     * @param line
     * @return
     */
    private boolean variableAssignmentChecker(String line) {
        Variable tempVariable;
        p = Pattern.compile(SECONDARY_VARIABLE_PARSER);
        m = p.matcher(line);
        m.matches();
        String firstVariable = m.group(GROUP_NAME);
        String value = m.group(SECONDARY_GROUP_VALUE);
        if (currentBlock.globalContains(firstVariable)) {
            if (!currentBlock.getGlobalVariable(firstVariable).isFinal()) {
                if (value.matches(VARIABLE_NAME)) {
                    if (currentBlock.globalContains(value)) {
                        if (Variable.assignable(currentBlock.getVariableType(firstVariable), (currentBlock
                                .getVariableType(value)))) {
                            return true;
                        }
                    }
                } else if (currentBlock.localContains(firstVariable)) {
                    if (currentBlock.getVariable(firstVariable).assignValue(value)) {
                        return true;
                    }
                } else if (currentBlock.globalContains(firstVariable)) {
                    tempVariable = currentBlock.getGlobalVariable(firstVariable);
                    currentBlock.addLocalVariable(VariableFactory.createVariable(getVariableNumber
                                    (tempVariable.variableGetTypeName()), tempVariable.getName(),
                            value, tempVariable.isFinal(), currentBlock));
                }
                return true;
            }
        }
        return false;
    }


    /**
     * A method which creates Variables out of  local scopes, and checks if a Variable exists with that same
     * scope.
     *
     * @param line
     * @throws VariableException
     */
    private void createVariables(String line) throws VariableException {
        int variableType;
        variableType = getVariableDeclarationType(line);
        boolean isFinal = containsFinal(line);
        String[] parsedVariables = parseVariables(line, variableType);
        for (String item : parsedVariables) {
            String name;
            String value;
            String[] nameAndValueSetter = getNameAndValue(item);
            name = nameAndValueSetter[NAME];
            value = nameAndValueSetter[VALUE];
            if (!currentBlock.addLocalVariable(VariableFactory.createVariable(variableType, name,
                    value, isFinal, currentBlock)
            )) {
                throw new VariableException();
            }
        }
    }


    /**
     * a Method which checks if a method call is valid.
     *
     * @param line
     * @param firstVariable
     * @param secondVariable
     * @throws IllegalLineException
     */
    private void methodCallCheck(String line, String firstVariable, String secondVariable)
            throws BlockException, IllegalLineException {
        // prep
        String[] methodArray = BlockFactory.getMethodCallGroups(line);
        String methodName = methodArray[METHOD_CALL], methodVariables = methodArray[METHOD_VARIABLES];
        String[] parsedParameters = methodVariables.split(",");
        if (methodVariables.equals("")) { //if the variables were empty
            parsedParameters = new String[0];
        }
        if (global.contains(methodName)) {
            theMethod = global.getMethod(methodName);
        }

        // compare parameter numbers
        if (!(theMethod.getParameterSize() == parsedParameters.length))
            throw new WrongMethodParameters();

        // compare parameter types in order.
        arrayCounter = 0;
        for (String parameter : parsedParameters) {
            // if the variable is a variable name.
            parameter.trim(); // take out any white spaces.
            firstVariable = theMethod.getParameters(arrayCounter).variableGetTypeName();
            if (parameter.matches(VARIABLE_NAME) && !parameter.equals(BlockFactory.TRUE) && !parameter.equals(BlockFactory.FALSE)) {
                if (currentBlock.globalContains(parameter)) {
                    if (currentBlock.getGlobalVariable(parameter).isAssigned())
                        secondVariable = currentBlock.getGlobalVariable(parameter).variableGetTypeName();
                    if (Variable.assignable(firstVariable, secondVariable)) {
                        arrayCounter++;
                        continue;
                    }
                    // the parameter is probably a constant, check validity and then compare type.
                }
                throw new IllegalLineException();
            } else {
                secondVariable = getValueType(parameter);
                if (!secondVariable.equals(ERROR)) {
                    if (Variable.assignable(firstVariable, secondVariable)) {
                        arrayCounter++;
                        continue;
                    }
                }
                throw new IllegalLineException();
            }

        }
    }


    /**
     * gets the value type of a value.
     * @param value
     * @return
     */
    public String getValueType(String value) {
        if (value.matches(SInt.INT_VALUE)) {
            return Variable.S_INT;
        }
        if (value.matches(SDouble.DOUBLE_VALUE)) {
            return Variable.S_DOUBLE;
        }
        if (value.matches(SString.STRING_VALUE)) {
            return Variable.S_STRING;
        }
        if (value.matches(SChar.CHAR_VALUE)) {
            return Variable.S_CHAR;
        }
        if (value.matches(SBoolean.BOOLEAN_VALUE)) {
            return Variable.S_BOOLEAN;
        }

        return ERROR;
    }


    /**
     * a Method which returns a number of a type which by using it can create a Variable.
     * @param variableType
     * @return
     */
    private int getVariableNumber(String variableType) {
        switch (variableType) {
            case Variable.S_INT:
                return INT_DEC_NUM;
            case Variable.S_DOUBLE:
                return DOUBLE_DEC_NUM;
            case Variable.S_STRING:
                return STRING_DEC_NUM;
            case Variable.S_CHAR:
                return CHAR_DEC_NUM;
            default:
                return BOOLEAN_DEC_NUM;

        }
    }


}




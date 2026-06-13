#include <iostream>
#include <vector>
#include <string>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <set>
#include <algorithm>

#include "filter_condition.h"

enum class TokenType {
    Identifier,
    Number,
    Operator,
    LogicalOp, 
    InOp,
    LeftBracket, 
    RightBracket,   
    Comma,          
    End             
};

struct Token {
    TokenType type;
    std::string lexeme;
};

class Condition {
public:
    virtual ~Condition() = default;
    virtual void print(int indent = 0) const = 0;
    virtual void trans(std::vector<FilterCondition>& conditions) const = 0;
};

class ComparisonCondition : public Condition {
public:
    std::string key;
    std::string op;
    double value;

    ComparisonCondition(const std::string& k, const std::string& o, double v)
        : key(k), op(o), value(v) {
    }

    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Comparison: " << key << " " << op << " " << value << std::endl;
    }

    void trans(std::vector<FilterCondition>& conditions) const override {
        std::set<float> value_set;
        value_set.insert(static_cast<float>(value));
        conditions.emplace_back(key, op, value_set);
    }
};

class InCondition : public Condition {
public:
    std::string key;
    std::set<double> values;

    InCondition(const std::string& k, const std::set<double>& vals)
        : key(k), values(vals) {
    }

    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "In: " << key << " IN [";
        bool first = true;
        for (double val : values) {
            if (!first) std::cout << ", ";
            std::cout << val;
            first = false;
        }
        std::cout << "]" << std::endl;
    }
    
    void trans(std::vector<FilterCondition>& conditions) const override {
        std::set<float> float_values;
        for (double val : values) {
            float_values.insert(static_cast<float>(val));
        }
        conditions.emplace_back(key, "IN", float_values);
    }
};

class LogicalCondition : public Condition {
public:
    std::string op;
    std::unique_ptr<Condition> left;
    std::unique_ptr<Condition> right;

    LogicalCondition(const std::string& o, std::unique_ptr<Condition> l, std::unique_ptr<Condition> r)
        : op(o), left(std::move(l)), right(std::move(r)) {
    }

    void print(int indent) const override {
        std::cout << std::string(indent, ' ') << "Logical: " << op << std::endl;
        left->print(indent + 4);
        right->print(indent + 4);
    }
    
    void trans(std::vector<FilterCondition>& conditions) const override {
        left->trans(conditions);
        right->trans(conditions);
    }
};

class Tokenizer {
public:
    Tokenizer(const std::string& input) : input_(input), index_(0) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (index_ < input_.size()) {
            if (std::isspace(static_cast<unsigned char>(input_[index_]))) {
                index_++;
                continue;
            }

            if (std::isalpha(static_cast<unsigned char>(input_[index_]))) {
                tokens.push_back(readIdentifier());
            }
            else if (std::isdigit(static_cast<unsigned char>(input_[index_])) || input_[index_] == '-') {
                tokens.push_back(readNumber());
            }
            else {
                switch (input_[index_]) {
                case '>':
                case '<':
                case '=':
                case '!':
                    tokens.push_back(readOperator());
                    break;
                case '[':
                    tokens.push_back({ TokenType::LeftBracket, "[" });
                    index_++;
                    break;
                case ']':
                    tokens.push_back({ TokenType::RightBracket, "]" });
                    index_++;
                    break;
                case ',':
                    tokens.push_back({ TokenType::Comma, "," });
                    index_++;
                    break;
                default:
                    throw std::runtime_error("Unknown character: " + std::string(1, input_[index_]));
                }
            }
        }
        tokens.push_back({ TokenType::End, "" });
        return tokens;
    }

private:
    Token readIdentifier() {
        size_t start = index_;
        while (index_ < input_.size() && (std::isalnum(static_cast<unsigned char>(input_[index_])) || input_[index_] == '_') ){
            index_++;
        }
        std::string id = input_.substr(start, index_ - start);

        if (id == "AND" || id == "OR") {
            return { TokenType::LogicalOp, id };
        }
        else if (id == "IN") {
            return { TokenType::InOp, id };
        }
        return { TokenType::Identifier, id };
    }

    Token readNumber() {
        size_t start = index_;
        if (input_[index_] == '-') {
            index_++;
        }
        while (index_ < input_.size() && (std::isdigit(static_cast<unsigned char>(input_[index_])) || input_[index_] == '.') ){
            index_++;
        }
        return { TokenType::Number, input_.substr(start, index_ - start) };
    }

    Token readOperator() {
        if (index_ + 1 < input_.size()) {
            char nextChar = input_[index_ + 1];
            if (nextChar == '=') {
                std::string op = std::string(1, input_[index_]) + "=";
                index_ += 2;
                return { TokenType::Operator, op };
            }
            else if (input_[index_] == '!' && nextChar == '=') {
                index_ += 2;
                return { TokenType::Operator, "!=" };
            }
        }
        std::string op(1, input_[index_]);
        index_++;
        return { TokenType::Operator, op };
    }

    std::string input_;
    size_t index_;
};

class Parser {
public:
    Parser(const std::vector<Token>& tokens) : tokens_(tokens), current_(0) {}

    std::unique_ptr<Condition> parse() {
        return parseExpression();
    }

private:
    std::unique_ptr<Condition> parseExpression() {
        auto left = parsePrimary();
        while (match(TokenType::LogicalOp)) {
            std::string op = previous().lexeme;
            auto right = parsePrimary();
            left = std::make_unique<LogicalCondition>(op, std::move(left), std::move(right));
        }
        return left;
    }

    std::unique_ptr<Condition> parsePrimary() {
        if (lookahead(0).type == TokenType::Identifier &&
            lookahead(1).type == TokenType::InOp) {
            return parseInCondition();
        }
        else {
            return parseComparisonCondition();
        }
    }

    std::unique_ptr<Condition> parseComparisonCondition() {
        Token key = consume(TokenType::Identifier, "Expected identifier");
        Token op = consume(TokenType::Operator, "Expected operator");
        Token val = consume(TokenType::Number, "Expected number");
        double value = std::stod(val.lexeme);
        return std::make_unique<ComparisonCondition>(key.lexeme, op.lexeme, value);
    }

    std::unique_ptr<Condition> parseInCondition() {
        Token key = consume(TokenType::Identifier, "Expected identifier");
        consume(TokenType::InOp, "Expected 'IN'");
        consume(TokenType::LeftBracket, "Expected '['");

        std::set<double> values;
        Token val = consume(TokenType::Number, "Expected number in list");
        values.insert(std::stod(val.lexeme));

        while (match(TokenType::Comma)) {
            Token nextVal = consume(TokenType::Number, "Expected number after comma");
            values.insert(std::stod(nextVal.lexeme));
        }

        consume(TokenType::RightBracket, "Expected ']' after value list");
        return std::make_unique<InCondition>(key.lexeme, values);
    }

    bool match(TokenType type) {
        if (check(type)) {
            advance();
            return true;
        }
        return false;
    }

    bool check(TokenType type) const {
        return !isAtEnd() && tokens_[current_].type == type;
    }

    Token advance() {
        if (!isAtEnd()) current_++;
        return previous();
    }

    bool isAtEnd() const {
        return tokens_[current_].type == TokenType::End;
    }

    Token previous() const {
        return tokens_[current_ - 1];
    }

    Token consume(TokenType type, const std::string& msg) {
        if (check(type)) return advance();
        throw std::runtime_error(msg);
    }

    Token lookahead(int offset) const {
        size_t pos = current_ + offset;
        return (pos < tokens_.size()) ? tokens_[pos] : Token{ TokenType::End, "" };
    }

    std::vector<Token> tokens_;
    size_t current_ = 0;
};
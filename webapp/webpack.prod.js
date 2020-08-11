const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyWebpackPlugin = require('copy-webpack-plugin');

module.exports = {
    // The path.resolve() function will create a path appending
    // the subfolders and the eventual filename.
    entry: {
        index: path.resolve(__dirname, "src", "index.js")
    },
    output: {
        path: path.resolve(__dirname, "dist")
    },
    plugins: [
        new HtmlWebpackPlugin({
            template: path.resolve(__dirname, "src", "index.html")
        }),
        new CopyWebpackPlugin({
            patterns: [{ from:'src/assets',to:'assets' }]
        }),
    ],
    module: {
        rules: [
            {
                test: /\.css$/,
                use: ["style-loader", "css-loader"]
            },
            {
                test: /\.(png|jpg)$/,
                loader: 'url-loader'
            },
            {
                test: /\.(png|svg|jpg|gif)$/,
                include : path.join(__dirname, 'assets'),
                loader: "file-loader?name=/assets/[name].[ext]",
            },
        ]
    }
}
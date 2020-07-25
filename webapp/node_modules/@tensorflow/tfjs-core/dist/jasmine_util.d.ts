/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { KernelBackend } from './backends/backend';
import { Environment, Flags } from './environment';
export declare type Constraints = {
    flags?: Flags;
    predicate?: (testEnv: TestEnv) => boolean;
};
export declare const NODE_ENVS: Constraints;
export declare const CHROME_ENVS: Constraints;
export declare const BROWSER_ENVS: Constraints;
export declare const SYNC_BACKEND_ENVS: Constraints;
export declare const HAS_WORKER: {
    predicate: () => boolean;
};
export declare const HAS_NODE_WORKER: {
    predicate: () => boolean;
};
export declare const ALL_ENVS: Constraints;
export declare function envSatisfiesConstraints(env: Environment, testEnv: TestEnv, constraints: Constraints): boolean;
export interface TestFilter {
    include?: string;
    startsWith?: string;
    excludes?: string[];
}
export declare function setupTestFilters(testFilters: TestFilter[], customInclude: (name: string) => boolean): void;
export declare function parseTestEnvFromKarmaFlags(args: string[], registeredTestEnvs: TestEnv[]): TestEnv;
export declare function describeWithFlags(name: string, constraints: Constraints, tests: (env: TestEnv) => void): void;
export interface TestEnv {
    name: string;
    backendName: string;
    flags?: Flags;
    isDataSync?: boolean;
}
export declare const TEST_ENVS: TestEnv[];
export declare function setTestEnvs(testEnvs: TestEnv[]): void;
export declare function registerTestEnv(testEnv: TestEnv): void;
export declare class TestKernelBackend extends KernelBackend {
    dispose(): void;
}
